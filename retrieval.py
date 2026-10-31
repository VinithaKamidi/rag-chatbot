"""Hybrid retrieval: dense ANN (HNSW) + BM25, fused with Reciprocal Rank Fusion,
then re-ranked by a cross-encoder (or a lexical fallback)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

import config
from rag.chunking import pages_label
from rag.embeddings import Embedder
from rag.metrics import tokenize
from rag.store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk_id: str
    text: str
    meta: dict
    dense_score: Optional[float] = None  # cosine similarity
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    fused_score: float = 0.0
    rerank_score: Optional[float] = None

    @property
    def filename(self) -> str:
        return self.meta["filename"]

    @property
    def pdf_id(self) -> str:
        return self.meta["pdf_id"]

    @property
    def page_start(self) -> int:
        return int(self.meta["page_start"])

    @property
    def page_end(self) -> int:
        return int(self.meta["page_end"])

    @property
    def pages(self) -> str:
        return pages_label(self.page_start, self.page_end)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    timings: dict
    best_dense: float
    candidates: int


class Reranker(Protocol):
    name: str

    def score(self, query: str, docs: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """MS MARCO MiniLM cross-encoder via fastembed (ONNX, CPU)."""

    def __init__(self, model_name: Optional[str] = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name or config.RERANK_MODEL
        self.model = TextCrossEncoder(model_name=self.name)

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, list(docs))]


class OverlapReranker:
    """Lexical fallback: share of distinct query terms present, plus a bigram bonus."""

    name = "lexical-overlap"

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        q = tokenize(query)
        qset = set(q)
        qbi = {(a, b) for a, b in zip(q, q[1:])}
        out = []
        for d in docs:
            toks = tokenize(d)
            dset = set(toks)
            dbi = {(a, b) for a, b in zip(toks, toks[1:])}
            term = len(qset & dset) / max(len(qset), 1)
            bi = len(qbi & dbi) / max(len(qbi), 1)
            out.append(term + 0.5 * bi)
        return out


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, reranker: Optional[Reranker] = None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._bm25: Optional[BM25Okapi] = None
        self._ids: list[str] = []
        self._pdf_of: np.ndarray = np.array([])
        self._built_for = -1

    # ------------------------------------------------------------- BM25 ----
    def _ensure_bm25(self) -> None:
        n = self.store.count()
        if self._bm25 is not None and self._built_for == n:
            return
        ids, pdfs, tokens = [], [], []
        for row in self.store.iter_chunks():
            ids.append(row["chunk_id"])
            pdfs.append(row["meta"]["pdf_id"])
            tokens.append(tokenize(row["text"]))
        self._ids, self._pdf_of = ids, np.array(pdfs)
        self._bm25 = BM25Okapi(tokens) if tokens else None
        self._built_for = n
        log.info("BM25 index built over %d chunks", n)

    def _bm25_top(self, query: str, k: int, pdf_ids: Optional[Sequence[str]]):
        self._ensure_bm25()
        qt = tokenize(query)
        if self._bm25 is None or not qt:
            return []
        scores = np.asarray(self._bm25.get_scores(qt))
        if pdf_ids:
            scores = np.where(np.isin(self._pdf_of, list(pdf_ids)), scores, -1.0)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top if scores[i] > 0]

    # ----------------------------------------------------------- search ----
    def search(
        self,
        query: str,
        *,
        hybrid: bool = True,
        rerank: bool = True,
        top_k: Optional[int] = None,
        pdf_ids: Optional[Sequence[str]] = None,
    ) -> RetrievalResult:
        top_k = top_k or config.TOP_K_FINAL
        t = {}
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        t["embed_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        dense = self.store.query(qvec, config.TOP_K_DENSE, pdf_ids)
        t["dense_ms"] = (time.perf_counter() - t1) * 1000

        cands: dict[str, Hit] = {}
        for rank, row in enumerate(dense, 1):
            cands[row["chunk_id"]] = Hit(
                chunk_id=row["chunk_id"], text=row["text"], meta=row["meta"],
                dense_score=row["score"], dense_rank=rank,
                fused_score=1.0 / (config.RRF_K + rank),
            )
        best_dense = dense[0]["score"] if dense else 0.0

        t2 = time.perf_counter()
        if hybrid:
            for rank, (cid, score) in enumerate(self._bm25_top(query, config.TOP_K_BM25, pdf_ids), 1):
                if cid not in cands:
                    row = self.store.get_chunk(cid)
                    if row is None:
                        continue
                    cands[cid] = Hit(chunk_id=cid, text=row["text"], meta=row["meta"])
                h = cands[cid]
                h.bm25_score, h.bm25_rank = score, rank
                h.fused_score += 1.0 / (config.RRF_K + rank)
        t["bm25_ms"] = (time.perf_counter() - t2) * 1000

        ordered = sorted(cands.values(), key=lambda h: h.fused_score, reverse=True)
        t3 = time.perf_counter()
        if rerank and self.reranker is not None and ordered:
            pool = ordered[: config.RERANK_CANDIDATES]
            for h, s in zip(pool, self.reranker.score(query, [h.text for h in pool])):
                h.rerank_score = s
            ordered = sorted(pool, key=lambda h: h.rerank_score, reverse=True)
        t["rerank_ms"] = (time.perf_counter() - t3) * 1000
        t["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(hits=ordered[:top_k], timings=t, best_dense=best_dense, candidates=len(cands))
