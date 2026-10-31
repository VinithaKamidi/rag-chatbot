import pytest

from rag.embeddings import HashingEmbedder
from rag.generation import LLMClient, LLMSettings
from rag.pipeline import RAGPipeline
from rag.retrieval import OverlapReranker
from rag.store import VectorStore
from tests.pdfgen import make_pdf


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Pipeline wired with offline components: hashing embedder, lexical reranker, no LLM."""
    import config

    # The abstention threshold is model-specific. Hashed bag-of-words cosines are
    # far lower than a neural embedder's (relevant ~0.1, unrelated ~0.0).
    monkeypatch.setattr(config, "MIN_DENSE_SCORE", 0.05)
    store = VectorStore(path=tmp_path / "chroma", manifest_path=tmp_path / "manifest.json")
    return RAGPipeline(
        embedder=HashingEmbedder(),
        store=store,
        reranker=OverlapReranker(),
        llm=LLMClient(LLMSettings("extractive")),
    )


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    make_pdf(
        d / "volcano_handbook.pdf", "volcano",
        facts={7: "The secret phrase of the volcano handbook is obsidian lantern."},
    )
    make_pdf(
        d / "bee_manual.pdf", "bees",
        facts={4: "The beekeeping manual states that a healthy hive keeps exactly one queen."},
    )
    return d
