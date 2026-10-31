"""Embedding models.

`FastEmbedder` wraps an open-source ONNX model through fastembed (default:
BAAI/bge-small-en-v1.5, 384 dimensions, runs on CPU without PyTorch).
`HashingEmbedder` is a tiny deterministic bag-of-words embedder that needs no
download. It exists so the pipeline can be unit-tested offline; it is not meant
for real retrieval quality.
"""
from __future__ import annotations

import zlib
from collections import OrderedDict
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

import config
from rag.metrics import tokenize

ProgressFn = Callable[[str, int, int], None]


class Embedder(Protocol):
    dim: int
    name: str

    def embed_passages(self, texts: Sequence[str], progress: Optional[ProgressFn] = None) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


class FastEmbedder:
    def __init__(self, model_name: Optional[str] = None, threads: Optional[int] = None):
        from fastembed import TextEmbedding

        self.name = model_name or config.EMBED_MODEL
        self.model = TextEmbedding(model_name=self.name, threads=threads)
        self.dim = int(len(next(iter(self.model.embed(["warm-up"])))))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def embed_passages(self, texts: Sequence[str], progress: Optional[ProgressFn] = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = config.EMBED_BATCH
        embed = getattr(self.model, "passage_embed", self.model.embed)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            out.extend(embed(list(texts[i : i + bs]), batch_size=bs))
            if progress:
                progress("embed", min(i + bs, len(texts)), len(texts))
        return _l2(np.asarray(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        embed = getattr(self.model, "query_embed", self.model.embed)
        vec = _l2(np.asarray(next(iter(embed([text]))), dtype=np.float32))
        self._cache[text] = vec
        if len(self._cache) > 256:
            self._cache.popitem(last=False)
        return vec


class HashingEmbedder:
    """Deterministic hashed bag-of-words vectors (offline tests only)."""

    name = "hashing-test-embedder"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):  # content words only, like any lexical embedder
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        return np.log1p(v)

    def embed_passages(self, texts: Sequence[str], progress: Optional[ProgressFn] = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = _l2(np.stack([self._vec(t) for t in texts]))
        if progress:
            progress("embed", len(texts), len(texts))
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text))
