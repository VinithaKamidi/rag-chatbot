"""End-to-end orchestration: ingestion (PDF -> vector DB) and question answering."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import config
from rag.chunking import chunk_pdf
from rag.embeddings import Embedder, FastEmbedder, HashingEmbedder
from rag.generation import (
    NO_ANSWER, LLMClient, LLMError, build_messages, build_sources, cited_numbers, extractive_answer,
)
from rag.ingest import extract_pdf, file_sha
from rag.retrieval import CrossEncoderReranker, Hit, OverlapReranker, Reranker, Retriever, RetrievalResult
from rag.store import VectorStore

log = logging.getLogger(__name__)
ProgressFn = Callable[[str, int, int], None]


@dataclass
class Answer:
    question: str
    text: str
    mode: str  # "llm" | "extractive" | "abstain"
    sources: list[dict]
    hits: list[Hit]
    timings: dict
    invalid_citations: list[int] = field(default_factory=list)
    note: str = ""

    @property
    def abstained(self) -> bool:
        return self.text.strip().startswith(NO_ANSWER[:30])

    @property
    def citations_valid(self) -> bool:
        return not self.invalid_citations


class RAGPipeline:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        reranker: Optional[Reranker] = None,
        llm: Optional[LLMClient] = None,
    ):
        self.store = store or VectorStore()
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self._retriever: Optional[Retriever] = None

    # models are created lazily so listing documents never loads an ONNX model
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = HashingEmbedder() if config.EMBEDDER_BACKEND == "hashing" else FastEmbedder()
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            try:
                self._reranker = CrossEncoderReranker()
            except Exception as exc:  # noqa: BLE001
                log.warning("Cross-encoder unavailable (%s); using lexical reranker", exc)
                self._reranker = OverlapReranker()
        return self._reranker

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.store, self.embedder, self.reranker)
        return self._retriever

    def warmup(self) -> dict:
        """Load models and build the BM25 index so the first real query is fast."""
        t0 = time.perf_counter()
        if self.store.count():
            self.retriever.search("warm up", rerank=True)
        else:
            _ = self.embedder, self.reranker
        return {"warmup_s": round(time.perf_counter() - t0, 2)}

    # ------------------------------------------------------------ ingest ---
    def ingest_pdf(self, path: str | Path, *, force: bool = False, ocr: Optional[bool] = None,
                   progress: Optional[ProgressFn] = None) -> dict:
        path = Path(path)
        progress = progress or (lambda *_: None)
        pdf_id = file_sha(path)
        if not force and self.store.has_pdf(pdf_id):
            return {"status": "skipped", "pdf_id": pdf_id, "filename": path.name,
                    **self.store.documents().get(pdf_id, {})}

        t0 = time.perf_counter()
        parsed = extract_pdf(path, ocr=ocr, progress=progress)
        t_extract = time.perf_counter() - t0

        t1 = time.perf_counter()
        chunks = chunk_pdf(parsed)
        t_chunk = time.perf_counter() - t1

        t2 = time.perf_counter()
        vectors = self.embedder.embed_passages([c.text for c in chunks], progress=progress)
        t_embed = time.perf_counter() - t2

        t3 = time.perf_counter()
        progress("index", 0, len(chunks))
        if force:
            self.store.col.delete(where={"pdf_id": pdf_id})
        self.store.add_chunks(chunks, vectors)
        t_index = time.perf_counter() - t3
        progress("index", len(chunks), len(chunks))

        info = {
            "pdf_id": pdf_id,
            "filename": path.name,
            "language": parsed.language,
            "chunks": len(chunks),
            "avg_chunk_words": round(sum(c.n_words for c in chunks) / max(len(chunks), 1)),
            "embed_model": getattr(self.embedder, "name", "unknown"),
            "seconds": {"extract_ocr": round(t_extract, 2), "chunk": round(t_chunk, 3),
                        "embed": round(t_embed, 2), "index": round(t_index, 2)},
            **parsed.stats,
        }
        self.store.record_doc(info)
        return {"status": "ingested", **info}

    def ingest_dir(self, directory: str | Path, *, force: bool = False, ocr: Optional[bool] = None,
                   progress: Optional[Callable[[str, dict], None]] = None) -> list[dict]:
        pdfs = sorted(Path(directory).glob("*.pdf"))
        results = []
        for i, pdf in enumerate(pdfs, 1):
            def inner(stage, done, total, _name=pdf.name):
                if progress:
                    progress("stage", {"file": _name, "stage": stage, "done": done, "total": total})
            res = self.ingest_pdf(pdf, force=force, ocr=ocr, progress=inner)
            results.append(res)
            if progress:
                progress("file_done", {"index": i, "of": len(pdfs), **res})
        return results

    # ------------------------------------------------------------ answer ---
    def retrieve(self, question: str, **kw) -> RetrievalResult:
        return self.retriever.search(question, **kw)

    def ask_stream(self, question: str, *, hybrid: bool = True, rerank: bool = True,
                   top_k: Optional[int] = None, pdf_ids: Optional[Sequence[str]] = None) -> Iterator[tuple]:
        """Yields ("retrieved", RetrievalResult), then ("token", str)..., then ("done", Answer)."""
        t0 = time.perf_counter()
        res = self.retrieve(question, hybrid=hybrid, rerank=rerank, top_k=top_k, pdf_ids=pdf_ids)
        yield "retrieved", res
        hits, timings = res.hits, dict(res.timings)
        note, mode, parts = "", "extractive", []

        def emit(text: str):
            if "ttft_ms" not in timings:
                timings["ttft_ms"] = (time.perf_counter() - t0) * 1000
            parts.append(text)
            return ("token", text)

        top_rerank = hits[0].rerank_score if hits else None
        weak_reason = ""
        if not hits:
            weak_reason = "no passages retrieved"
        elif res.best_dense < config.MIN_DENSE_SCORE:
            weak_reason = f"best dense similarity {res.best_dense:.2f} < {config.MIN_DENSE_SCORE:.2f}"
        elif config.MIN_RERANK_SCORE is not None and top_rerank is not None and top_rerank < config.MIN_RERANK_SCORE:
            weak_reason = f"top rerank score {top_rerank:.2f} < {config.MIN_RERANK_SCORE:.2f}"

        if weak_reason:
            mode = "abstain"
            note = f"{weak_reason}; LLM skipped"
            yield emit(NO_ANSWER)
        else:
            t_gen = time.perf_counter()
            used_llm = False
            if self.llm.available:
                try:
                    for tok in self.llm.stream(build_messages(question, hits)):
                        used_llm = True
                        yield emit(tok)
                    mode = "llm"
                except LLMError as exc:
                    note = f"LLM unavailable ({exc}); used extractive fallback"
                    log.warning(note)
                    if used_llm:  # partial stream already shown; keep it and flag it
                        mode = "llm"
                        note += " (answer may be truncated)"
            if mode != "llm":
                mode = "extractive"
                parts.clear()
                timings.pop("ttft_ms", None)
                yield emit(extractive_answer(question, hits))
            timings["generate_ms"] = (time.perf_counter() - t_gen) * 1000

        text = "".join(parts).strip()
        valid_n = len(hits)
        nums = cited_numbers(text)
        invalid = [n for n in nums if not 1 <= n <= valid_n]
        if mode == "llm" and not nums and not text.startswith(NO_ANSWER[:30]):
            note = (note + "; " if note else "") + "answer contains no citations"
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        yield "done", Answer(question, text, mode, build_sources(hits, text), hits, timings, invalid, note)

    def ask(self, question: str, **kw) -> Answer:
        answer = None
        for kind, payload in self.ask_stream(question, **kw):
            if kind == "done":
                answer = payload
        assert answer is not None
        return answer
