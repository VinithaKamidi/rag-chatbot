"""Vector store: ChromaDB (open source, persistent) with an HNSW index.

Chroma keeps vectors, chunk text and metadata together on disk. A small JSON
manifest next to it records one summary row per ingested PDF so the UI can list
the corpus without scanning every chunk.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator, Optional, Sequence

import chromadb
import numpy as np

import config
from rag.chunking import Chunk

# HNSW parameters: M and construction_ef trade build time/memory for recall;
# search_ef trades query latency for recall. These are good CPU defaults.
HNSW_SETTINGS = {
    "hnsw:space": "cosine",
    "hnsw:M": 32,
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 64,
}


class VectorStore:
    def __init__(self, path: Optional[Path] = None, collection: Optional[str] = None,
                 manifest_path: Optional[Path] = None):
        self.path = Path(path or config.CHROMA_DIR)
        self.manifest_path = Path(manifest_path or config.MANIFEST_PATH)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.col = self.client.get_or_create_collection(
            name=collection or config.COLLECTION, metadata=dict(HNSW_SETTINGS)
        )

    # ------------------------------------------------------------ writing --
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: np.ndarray, batch: int = 1000) -> None:
        assert len(chunks) == len(embeddings), "one embedding per chunk"
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self.col.upsert(  # upsert => re-ingesting the same PDF is idempotent
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[i : i + batch].tolist(),
                documents=[c.text for c in part],
                metadatas=[c.metadata() for c in part],
            )

    def delete_pdf(self, pdf_id: str) -> None:
        self.col.delete(where={"pdf_id": pdf_id})
        manifest = self.documents()
        manifest.pop(pdf_id, None)
        self._write_manifest(manifest)

    # ------------------------------------------------------------ reading --
    def count(self) -> int:
        return self.col.count()

    def query(self, embedding: np.ndarray, k: int, pdf_ids: Optional[Sequence[str]] = None) -> list[dict]:
        n = self.count()
        if n == 0:
            return []
        where = None
        if pdf_ids:
            where = {"pdf_id": pdf_ids[0]} if len(pdf_ids) == 1 else {"pdf_id": {"$in": list(pdf_ids)}}
        res = self.col.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, n),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"chunk_id": cid, "text": doc, "meta": meta, "score": 1.0 - float(dist)}  # cosine similarity
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def iter_chunks(self, page_size: int = 2000) -> Iterator[dict]:
        offset = 0
        while True:
            res = self.col.get(limit=page_size, offset=offset, include=["documents", "metadatas"])
            if not res["ids"]:
                return
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                yield {"chunk_id": cid, "text": doc, "meta": meta}
            offset += page_size

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        res = self.col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        return {"chunk_id": chunk_id, "text": res["documents"][0], "meta": res["metadatas"][0]}

    # ----------------------------------------------------------- manifest --
    def documents(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError:
            return {}

    def has_pdf(self, pdf_id: str) -> bool:
        if pdf_id not in self.documents():
            return False
        return bool(self.col.get(where={"pdf_id": pdf_id}, limit=1)["ids"])

    def record_doc(self, info: dict) -> None:
        manifest = self.documents()
        manifest[info["pdf_id"]] = {**info, "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._write_manifest(manifest)

    def _write_manifest(self, manifest: dict) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(tmp, self.manifest_path)
