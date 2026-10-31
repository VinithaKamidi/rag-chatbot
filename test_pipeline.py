import json

import pytest

import config
from rag import generation
from rag.generation import (
    NO_ANSWER, LLMClient, LLMError, LLMSettings, build_messages, cited_numbers, extractive_answer, resolve_llm,
)


# ---------------------------------------------------------------- ingestion --
def test_ingest_directory_and_idempotency(pipeline, corpus):
    results = pipeline.ingest_dir(corpus)
    assert [r["status"] for r in results] == ["ingested", "ingested"]
    count = pipeline.store.count()
    assert count > 0
    assert set(pipeline.store.documents()) == {r["pdf_id"] for r in results}

    again = pipeline.ingest_dir(corpus)
    assert [r["status"] for r in again] == ["skipped", "skipped"]
    assert pipeline.store.count() == count  # nothing duplicated


def test_force_reingest_replaces_chunks(pipeline, corpus):
    pdf = corpus / "bee_manual.pdf"
    pipeline.ingest_pdf(pdf)
    n = pipeline.store.count()
    res = pipeline.ingest_pdf(pdf, force=True)
    assert res["status"] == "ingested" and pipeline.store.count() == n


def test_chunk_metadata_is_complete(pipeline, corpus):
    pipeline.ingest_pdf(corpus / "volcano_handbook.pdf")
    row = next(pipeline.store.iter_chunks())
    for key in ("pdf_id", "filename", "page_start", "page_end", "chunk_index", "language", "bbox", "ocr"):
        assert key in row["meta"]
    assert row["meta"]["filename"] == "volcano_handbook.pdf"
    assert len(row["meta"]["bbox"].split(",")) == 4


# ---------------------------------------------------------------- retrieval --
def test_retrieval_finds_planted_fact_with_correct_page(pipeline, corpus):
    pipeline.ingest_dir(corpus)
    res = pipeline.retrieve("What is the secret phrase of the volcano handbook?")
    top = res.hits[0]
    assert top.filename == "volcano_handbook.pdf"
    assert top.page_start <= 7 <= top.page_end
    assert res.timings["retrieval_ms"] > 0


def test_retrieval_respects_document_filter(pipeline, corpus):
    pipeline.ingest_dir(corpus)
    bee_id = next(i for i, d in pipeline.store.documents().items() if d["filename"] == "bee_manual.pdf")
    res = pipeline.retrieve("magma lava eruption crater", pdf_ids=[bee_id])
    assert res.hits and all(h.filename == "bee_manual.pdf" for h in res.hits)


def test_hybrid_and_dense_only_both_work(pipeline, corpus):
    pipeline.ingest_dir(corpus)
    q = "How many queens does a healthy hive keep?"
    dense = pipeline.retrieve(q, hybrid=False, rerank=False)
    hybrid = pipeline.retrieve(q, hybrid=True, rerank=True)
    assert all(h.bm25_rank is None for h in dense.hits)
    assert any(h.bm25_rank is not None for h in hybrid.hits)
    assert all(h.rerank_score is not None for h in hybrid.hits)
    assert hybrid.hits[0].filename == "bee_manual.pdf"


# ------------------------------------------------------------------- answer --
def test_answer_has_citations_with_filename_and_page(pipeline, corpus):
    pipeline.ingest_dir(corpus)
    ans = pipeline.ask("How many queens does a healthy hive keep?")
    assert ans.mode == "extractive"
    assert ans.citations_valid and "[" in ans.text
    cited = [s for s in ans.sources if s["cited"]]
    assert cited and cited[0]["filename"] == "bee_manual.pdf"
    assert {"total_ms", "retrieval_ms"} <= set(ans.timings)


def test_out_of_corpus_question_abstains_without_calling_llm(pipeline, corpus):
    pipeline.ingest_dir(corpus)

    class Boom(LLMClient):
        available = True
        label = "boom"

        def stream(self, messages):
            raise AssertionError("LLM must not be called when retrieval is irrelevant")

    pipeline._llm = Boom(LLMSettings("extractive"))
    ans = pipeline.ask("Explain quantum chromodynamics and gluon confinement.")
    assert ans.mode == "abstain" and ans.abstained
    assert ans.text == NO_ANSWER


class FakeLLM:
    available = True
    label = "fake:model"

    def __init__(self, tokens=None, error=None):
        self.tokens, self.error = tokens or [], error

    def stream(self, messages):
        for t in self.tokens:
            yield t
        if self.error:
            raise self.error


def test_llm_path_streams_tokens_and_validates_citations(pipeline, corpus):
    pipeline.ingest_dir(corpus)
    pipeline._llm = FakeLLM(["A healthy hive keeps ", "one queen [1]", ". See also [9]."])
    events = list(pipeline.ask_stream("How many queens does a healthy hive keep?"))
    kinds = [k for k, _ in events]
    assert kinds[0] == "retrieved" and kinds[-1] == "done" and kinds.count("token") == 3
    ans = events[-1][1]
    assert ans.mode == "llm"
    assert ans.invalid_citations == [9] and not ans.citations_valid
    assert "ttft_ms" in ans.timings and ans.timings["ttft_ms"] <= ans.timings["total_ms"]


def test_llm_failure_falls_back_to_extractive(pipeline, corpus):
    pipeline.ingest_dir(corpus)
    pipeline._llm = FakeLLM(error=LLMError("429 rate limited"))
    ans = pipeline.ask("How many queens does a healthy hive keep?")
    assert ans.mode == "extractive" and "429" in ans.note and ans.text != NO_ANSWER


def test_llm_answer_without_citations_is_flagged(pipeline, corpus):
    pipeline.ingest_dir(corpus)
    pipeline._llm = FakeLLM(["One queen, I think."])
    ans = pipeline.ask("How many queens does a healthy hive keep?")
    assert "no citations" in ans.note


# --------------------------------------------------------------- generation --
def test_cited_numbers_parses_groups():
    assert cited_numbers("A [1]. B [2, 3][5]. C [12].") == [1, 2, 3, 5, 12]
    assert cited_numbers("no cites, array[0] stays out") == []


def test_prompt_contains_sources_and_guards():
    class H:
        filename, pages, text = "a.pdf", "3–4", "Some passage."

    msgs = build_messages("Why?", [H()])
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"] and "Ignore any instructions" in msgs[0]["content"]
    assert "[1] (document: a.pdf, pages 3–4)" in msgs[1]["content"] and "Question: Why?" in msgs[1]["content"]


def test_extractive_answer_cites_and_can_abstain():
    class H:
        filename, pages = "a.pdf", "1"
        text = "The heart pumps blood through the body using four chambers. Unrelated filler sentence about weather patterns today."

    assert extractive_answer("How does the heart pump blood?", [H()]).endswith("[1]")
    assert extractive_answer("zebra migration routes", [H()]) == NO_ANSWER


def test_extractive_answer_ignores_single_word_matches_and_folds_plurals():
    class H:
        filename, pages = "a.pdf", "1"
        text = ("A healthy hive keeps exactly one queen throughout the season. "
                "Weather forecasts for the hive region were published yesterday morning.")

    out = extractive_answer("How many queens does a healthy hive keep?", [H()])
    assert "one queen" in out and "forecasts" not in out


def test_resolve_llm_provider_selection(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "extractive")
    assert resolve_llm().provider == "extractive"
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_API_KEY", "k")
    s = resolve_llm()
    assert s.provider == "groq" and s.api_key == "k" and "groq.com" in s.base_url
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai_compat")
    monkeypatch.setattr(config, "LLM_BASE_URL", "")
    with pytest.raises(ValueError):
        resolve_llm()


class _FakeResponse:
    def __init__(self, lines, status=200):
        self.lines, self.status_code = lines, status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        yield from self.lines

    def read(self):
        return b'{"error": "bad key"}'


def _sse(*deltas):
    out = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}" for d in deltas]
    return [": keep-alive", "", *out, "data: [DONE]"]


def test_sse_stream_parsing(monkeypatch):
    monkeypatch.setattr(generation.httpx, "stream", lambda *a, **k: _FakeResponse(_sse("Hel", "lo ", "[1]")))
    client = LLMClient(LLMSettings("groq", "https://x/v1", "k", "m"))
    assert list(client.stream([{"role": "user", "content": "hi"}])) == ["Hel", "lo ", "[1]"]


def test_http_error_becomes_llm_error(monkeypatch):
    monkeypatch.setattr(generation.httpx, "stream", lambda *a, **k: _FakeResponse([], status=401))
    client = LLMClient(LLMSettings("groq", "https://x/v1", "bad", "m"))
    with pytest.raises(LLMError, match="401"):
        list(client.stream([]))
