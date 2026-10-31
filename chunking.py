"""Deterministic, page-aware chunking.

Pages are flattened into one word stream (remembering each word's page) and
cut into overlapping windows. Windows prefer to end on a sentence boundary
within the last 15% of the window. The same input always produces the same
chunks and chunk ids, which keeps embeddings reproducible.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

import config
from rag.ingest import ParsedPDF

_SENT_END = re.compile(r"[.!?][\"')\]]?$")


@dataclass
class Chunk:
    chunk_id: str
    pdf_id: str
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    n_words: int
    language: str
    bbox: str  # "x0,y0,x1,y1": text region of the chunk's first page (PDF points)
    ocr: bool

    def metadata(self) -> dict:
        """Scalar-only metadata, as required by the vector store."""
        d = asdict(self)
        d.pop("text")
        return d

    @property
    def pages(self) -> str:
        return pages_label(self.page_start, self.page_end)


def pages_label(start: int, end: int) -> str:
    return f"{start}" if start == end else f"{start}–{end}"


def chunk_pdf(
    parsed: ParsedPDF,
    *,
    chunk_words: Optional[int] = None,
    overlap: Optional[float] = None,
    min_words: Optional[int] = None,
) -> list[Chunk]:
    chunk_words = chunk_words or config.CHUNK_WORDS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    min_words = config.CHUNK_MIN_WORDS if min_words is None else min_words
    overlap_words = int(round(chunk_words * overlap))

    words: list[str] = []
    word_page: list[int] = []
    for p in parsed.pages:
        toks = p.text.split()
        words.extend(toks)
        word_page.extend([p.number] * len(toks))
    total = len(words)
    by_num = {p.number: p for p in parsed.pages}

    chunks: list[Chunk] = []
    start = 0
    while start < total:
        end = min(start + chunk_words, total)
        if end < total:
            floor = start + int(chunk_words * 0.85)
            for j in range(end, floor, -1):
                if _SENT_END.search(words[j - 1]):
                    end = j
                    break
            if total - end < min_words:  # do not leave a tiny orphan tail
                end = total

        if end - start >= min_words or not chunks:
            ps, pe = word_page[start], word_page[end - 1]
            first = by_num[ps]
            chunks.append(
                Chunk(
                    chunk_id=f"{parsed.pdf_id}:{len(chunks):05d}",
                    pdf_id=parsed.pdf_id,
                    filename=parsed.filename,
                    chunk_index=len(chunks),
                    page_start=ps,
                    page_end=pe,
                    text=" ".join(words[start:end]),
                    n_words=end - start,
                    language=first.language,
                    bbox=",".join(f"{v:.1f}" for v in first.bbox),
                    ocr=any(by_num[n].ocr for n in range(ps, pe + 1)),
                )
            )
        if end >= total:
            break
        start = max(end - overlap_words, start + 1)
    return chunks
