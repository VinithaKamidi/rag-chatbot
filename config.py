"""Central configuration.

Every value can be overridden with an environment variable, so the same code
runs on a laptop, in Docker, or on a hosted demo without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE per line) so secrets can stay out of git.

    Real environment variables always win over the file.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


load_dotenv(ROOT / ".env")


def _get(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


# ----------------------------------------------------------------- storage --
DATA_DIR = Path(_get("RAG_DATA_DIR", ROOT / "data", Path))
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
MANIFEST_PATH = DATA_DIR / "manifest.json"
COLLECTION = _get("RAG_COLLECTION", "pdf_chunks")

# --------------------------------------------------------------- ingestion --
OCR_ENABLED = _get("OCR_ENABLED", True, bool)
OCR_LANG = _get("OCR_LANG", "eng")  # e.g. "eng+hin+tel" if those packs are installed
OCR_DPI = _get("OCR_DPI", 200, int)
OCR_WORKERS = _get("OCR_WORKERS", max(1, min(4, os.cpu_count() or 1)), int)
OCR_MIN_CHARS = _get("OCR_MIN_CHARS", 40, int)  # fewer native chars => page is "scanned"
TESSERACT_CMD = _get("TESSERACT_CMD", "")  # set on Windows if tesseract is not on PATH
HEADER_FOOTER_EDGE_LINES = _get("HEADER_FOOTER_EDGE_LINES", 3, int)
HEADER_FOOTER_MIN_RATIO = _get("HEADER_FOOTER_MIN_RATIO", 0.4, float)

# ---------------------------------------------------------------- chunking --
# 375 words is roughly 500 tokens for English prose. It is also close to the
# 512-token input limit of the default embedding model, so going larger would
# silently truncate the end of every chunk.
CHUNK_WORDS = _get("CHUNK_WORDS", 375, int)
CHUNK_OVERLAP = _get("CHUNK_OVERLAP", 0.20, float)  # fraction of CHUNK_WORDS
CHUNK_MIN_WORDS = _get("CHUNK_MIN_WORDS", 40, int)

# --------------------------------------------------------------- embedding --
EMBED_MODEL = _get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384-dim, ONNX, CPU-fast
EMBED_BATCH = _get("EMBED_BATCH", 64, int)
# "fastembed" (real model, the default) or "hashing" (no download, poor quality).
# "hashing" exists only so the UI and tests can run offline; never use it for the demo.
EMBEDDER_BACKEND = _get("RAG_EMBEDDER", "fastembed")

# --------------------------------------------------------------- retrieval --
TOP_K_DENSE = _get("TOP_K_DENSE", 30, int)
TOP_K_BM25 = _get("TOP_K_BM25", 30, int)
RERANK_CANDIDATES = _get("RERANK_CANDIDATES", 15, int)
TOP_K_FINAL = _get("TOP_K_FINAL", 5, int)
RRF_K = _get("RRF_K", 60, int)
RERANK_MODEL = _get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
# Abstention guards: if retrieval is clearly irrelevant we skip the LLM and say
# so. Both values are specific to the embedding/reranker models in use, so run
# `python -m scripts.evaluate --calibrate` on your corpus and set them from the
# report. The defaults are deliberately permissive (the dense guard rarely fires
# for BGE models, whose similarity range is compressed; the rerank guard is off).
MIN_DENSE_SCORE = _get("MIN_DENSE_SCORE", 0.30, float)
MIN_RERANK_SCORE = _get("MIN_RERANK_SCORE", None, float)  # cross-encoder logit; None = disabled

# -------------------------------------------------------------- generation --
LLM_PROVIDER = _get("LLM_PROVIDER", "auto")  # auto | groq | ollama | openai_compat | extractive
LLM_MODEL = _get("LLM_MODEL", "")
LLM_BASE_URL = _get("LLM_BASE_URL", "")
LLM_API_KEY = _get("LLM_API_KEY", "")
LLM_MAX_TOKENS = _get("LLM_MAX_TOKENS", 350, int)
LLM_TEMPERATURE = _get("LLM_TEMPERATURE", 0.1, float)
LLM_TIMEOUT = _get("LLM_TIMEOUT", 20.0, float)

LATENCY_TARGET_S = (2.0, 5.0)  # the brief's end-to-end budget
