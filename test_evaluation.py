import json

from scripts.build_eval_set import build, eligible, sample_chunks
from scripts.evaluate import best_threshold, run_evaluation, to_markdown
from rag.metrics import tokenize


def test_best_threshold_separates_clean_classes():
    out = best_threshold(pos=[0.6, 0.7, 0.8], neg=[0.1, 0.2, 0.3])
    assert 0.3 < out["threshold"] <= 0.6 and out["balanced_accuracy"] == 1.0


def test_best_threshold_reports_overlap_honestly():
    out = best_threshold(pos=[0.2, 0.5, 0.8], neg=[0.1, 0.4, 0.6])
    assert out["balanced_accuracy"] < 1.0


def test_eval_set_is_seeded_balanced_and_deterministic(pipeline, corpus):
    pipeline.ingest_dir(corpus)
    rows = list(pipeline.store.iter_chunks())
    a = sample_chunks(rows, 6, seed=1)
    b = sample_chunks(rows, 6, seed=1)
    assert [r["chunk_id"] for r in a] == [r["chunk_id"] for r in b]
    assert len({r["meta"]["pdf_id"] for r in a[:2]}) == 2  # round-robin across documents
    assert all(eligible(r) for r in a)


def test_full_evaluation_run_produces_consistent_report(pipeline, corpus, tmp_path):
    pipeline.ingest_dir(corpus)
    questions = build(pipeline, n=8, seed=3, keyword=True)
    assert questions and all(q["gold_page_start"] <= q["gold_page_end"] for q in questions)
    assert all(set(tokenize(q["question"])) for q in questions)

    res = run_evaluation(
        pipeline, questions, ooc=["Explain quantum chromodynamics gluon confinement", "Neapolitan margherita pizza recipe"],
        n_answers=4, calibrate=True,
    )
    r = res["retrieval"]
    assert set(r) == {"dense", "hybrid", "hybrid+rerank"}
    for cfg in r.values():
        assert cfg["recall@1"] <= cfg["recall@3"] <= cfg["recall@5"] <= cfg["recall@10"] <= 1.0
        assert 0 < cfg["mrr"] <= 1.0
    assert r["hybrid"]["recall@10"] >= 0.6  # keyword queries from a chunk should find its pages
    assert res["answers"]["ooc_abstention_rate"] == 1.0
    assert 0.0 <= res["answers"]["hallucination_rate_proxy"] <= 1.0
    assert res["latency"]["total_p95_ms"] >= res["latency"]["total_p50_ms"]
    assert "dense" in res["calibration"]

    json.dumps(res)  # must be serialisable for the app's Evaluation tab
    md = to_markdown(res)
    assert "## Retrieval" in md and "hybrid+rerank" in md
