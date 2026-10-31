"""Evaluate retrieval quality, latency, citations and hallucination.

    python -m scripts.evaluate                     # uses eval/eval_set.jsonl
    python -m scripts.evaluate --calibrate         # also suggests abstention thresholds
    python -m scripts.evaluate --llm-judge         # extra, LLM-based faithfulness check
    python -m scripts.evaluate --sleep 2           # pause between LLM calls (rate limits)

Writes eval/results.json (shown in the app's Evaluation tab) and eval/results.md.

What is measured
  Retrieval   Recall@1/3/5/10 and MRR for dense-only, hybrid and hybrid+rerank. A hit is
              "relevant" when it is from the gold document and its page range overlaps
              the gold page range (page-level, which is what citations expose).
              chunk_recall@5 is the stricter exact-chunk version.
  Latency     End-to-end p50/p95 over full question->answer runs, plus retrieval-only p95.
  Answers     citation_validity  answers whose [n] all point at provided sources
              citation_hit       answers citing a source on the gold page
              hallucination_rate_proxy  answers with a sentence not lexically supported by
                                        the retrieved text (a screen, not ground truth)
              ooc_abstention_rate       out-of-corpus questions correctly refused
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import config
from rag.generation import NO_ANSWER
from rag.metrics import groundedness, mrr, pages_overlap, percentile, recall_at_k
from rag.pipeline import RAGPipeline

EVAL_DIR = config.ROOT / "eval"
CONFIGS = {
    "dense": {"hybrid": False, "rerank": False},
    "hybrid": {"hybrid": True, "rerank": False},
    "hybrid+rerank": {"hybrid": True, "rerank": True},
}

JUDGE_PROMPT = (
    "You are a strict fact-checker. Sources and an answer follow.\n"
    "Reply with exactly one word: SUPPORTED if every factual claim in the answer is supported by the sources, "
    "otherwise UNSUPPORTED.\n\nSources:\n{sources}\n\nAnswer:\n{answer}"
)


def is_relevant(hit, gold: dict) -> bool:
    return hit.filename == gold["gold_filename"] and pages_overlap(
        hit.page_start, hit.page_end, int(gold["gold_page_start"]), int(gold["gold_page_end"])
    )


def first_rank(hits, pred) -> Optional[int]:
    return next((i for i, h in enumerate(hits, 1) if pred(h)), None)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def best_threshold(pos: Sequence[float], neg: Sequence[float]) -> dict:
    """Threshold maximising balanced accuracy for 'answerable (pos) vs out-of-corpus (neg)'."""
    best = {"threshold": None, "balanced_accuracy": 0.0}
    for t in sorted(set(pos) | set(neg)):
        tpr = sum(p >= t for p in pos) / len(pos)  # answerable kept
        tnr = sum(n < t for n in neg) / len(neg)  # out-of-corpus rejected
        ba = (tpr + tnr) / 2
        if ba > best["balanced_accuracy"]:
            best = {"threshold": round(float(t), 4), "balanced_accuracy": round(ba, 3),
                    "answerable_kept": round(tpr, 3), "ooc_rejected": round(tnr, 3)}
    return best


def evaluate_retrieval(pipe: RAGPipeline, rows: list[dict]) -> tuple[dict, dict]:
    report, signals = {}, {"dense": [], "rerank": [], "retrieval_ms": []}
    for name, flags in CONFIGS.items():
        page_ranks, chunk_ranks, lat = [], [], []
        for row in rows:
            res = pipe.retrieve(row["question"], top_k=10, **flags)
            page_ranks.append(first_rank(res.hits, lambda h: is_relevant(h, row)))
            chunk_ranks.append(first_rank(res.hits, lambda h: h.chunk_id == row.get("gold_chunk_id")))
            lat.append(res.timings["retrieval_ms"])
            if name == "hybrid+rerank":
                signals["dense"].append(res.best_dense)
                signals["rerank"].append(res.hits[0].rerank_score if res.hits and res.hits[0].rerank_score is not None else float("nan"))
        report[name] = {
            **{f"recall@{k}": recall_at_k(page_ranks, k) for k in (1, 3, 5, 10)},
            "mrr": mrr(page_ranks),
            "chunk_recall@5": recall_at_k(chunk_ranks, 5),
            "retrieval_p50_ms": percentile(lat, 50),
            "retrieval_p95_ms": percentile(lat, 95),
        }
    return report, signals


def evaluate_answers(pipe: RAGPipeline, rows: list[dict], ooc: list[str], n: int, judge: bool, sleep: float):
    totals, ttft, retr = [], [], []
    valid = hit = halluc = abstained = judged = judge_bad = 0
    answered = 0
    subset = rows[:n]
    for row in subset:
        ans = pipe.ask(row["question"])
        totals.append(ans.timings["total_ms"])
        retr.append(ans.timings["retrieval_ms"])
        if "ttft_ms" in ans.timings:
            ttft.append(ans.timings["ttft_ms"])
        if ans.abstained:
            abstained += 1
        else:
            answered += 1
            valid += int(ans.citations_valid and any(s["cited"] for s in ans.sources))
            hit += int(any(s["cited"] and s["filename"] == row["gold_filename"] and pages_overlap(
                s["page_start"], s["page_end"], int(row["gold_page_start"]), int(row["gold_page_end"]))
                for s in ans.sources))
            frac, _ = groundedness(ans.text, [h.text for h in ans.hits])
            halluc += int(frac < 1.0)
            if judge and pipe.llm.available:
                src = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(ans.hits, 1))
                verdict = pipe.llm.complete([{"role": "user", "content": JUDGE_PROMPT.format(sources=src, answer=ans.text)}])
                judged += 1
                judge_bad += int(verdict.strip().upper().startswith("UNSUPPORTED"))
        if sleep:
            time.sleep(sleep)

    ooc_refused = 0
    for q in ooc:
        ooc_refused += int(pipe.ask(q).abstained)
        if sleep:
            time.sleep(sleep)

    a = max(answered, 1)
    answers = {
        "answered": answered,
        "abstained_in_corpus_rate": abstained / max(len(subset), 1),
        "citation_validity": valid / a,
        "citation_hit": hit / a,
        "hallucination_rate_proxy": halluc / a,
        "ooc_abstention_rate": ooc_refused / max(len(ooc), 1),
    }
    if judged:
        answers["hallucination_rate_llm_judge"] = judge_bad / judged
    latency = {
        "n": len(totals),
        "total_p50_ms": percentile(totals, 50),
        "total_p95_ms": percentile(totals, 95),
        "total_mean_ms": sum(totals) / max(len(totals), 1),
        "ttft_p50_ms": percentile(ttft, 50),
        "retrieval_p95_ms": percentile(retr, 95),
        "within_5s_fraction": sum(t <= config.LATENCY_TARGET_S[1] * 1000 for t in totals) / max(len(totals), 1),
    }
    return answers, latency


def calibration(pipe: RAGPipeline, signals: dict, ooc: list[str]) -> dict:
    neg_dense, neg_rerank = [], []
    for q in ooc:
        res = pipe.retrieve(q, top_k=5)
        neg_dense.append(res.best_dense)
        if res.hits and res.hits[0].rerank_score is not None:
            neg_rerank.append(res.hits[0].rerank_score)
    out = {"dense": best_threshold(signals["dense"], neg_dense) if neg_dense else None}
    pos_rr = [x for x in signals["rerank"] if x == x]  # drop NaN
    if pos_rr and neg_rerank:
        out["rerank"] = best_threshold(pos_rr, neg_rerank)
    out["how_to_apply"] = "export MIN_DENSE_SCORE=<dense.threshold> MIN_RERANK_SCORE=<rerank.threshold>"
    return out


def run_evaluation(pipe: RAGPipeline, rows: list[dict], ooc: list[str], *, n_answers: int = 50,
                   judge: bool = False, calibrate: bool = False, sleep: float = 0.0) -> dict:
    retrieval, signals = evaluate_retrieval(pipe, rows)
    answers, latency = evaluate_answers(pipe, rows, ooc, n_answers, judge, sleep)
    result = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"), "n_questions": len(rows), "n_answered": min(n_answers, len(rows)),
            "n_ooc": len(ooc), "question_kinds": sorted({r.get("kind", "manual") for r in rows}),
            "embed_model": getattr(pipe.embedder, "name", "?"), "reranker": pipe.reranker.name,
            "llm": pipe.llm.label, "chunks": pipe.store.count(), "documents": len(pipe.store.documents()),
            "min_dense_score": config.MIN_DENSE_SCORE, "min_rerank_score": config.MIN_RERANK_SCORE,
        },
        "retrieval": retrieval, "latency": latency, "answers": answers,
    }
    if calibrate:
        result["calibration"] = calibration(pipe, signals, ooc)
    return result


def to_markdown(res: dict) -> str:
    m = res["meta"]
    lines = [f"# Evaluation report ({m['date']})", "",
             f"{m['n_questions']} questions ({', '.join(m['question_kinds'])}) over {m['documents']} documents / "
             f"{m['chunks']} chunks. Embeddings `{m['embed_model']}`, reranker `{m['reranker']}`, LLM `{m['llm']}`.", "",
             "## Retrieval (page-level match)", "",
             "| config | R@1 | R@3 | R@5 | R@10 | MRR | chunk R@5 | retrieval p95 (ms) |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res["retrieval"].items():
        lines.append(f"| {name} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['recall@10']:.3f} | "
                     f"{r['mrr']:.3f} | {r['chunk_recall@5']:.3f} | {r['retrieval_p95_ms']:.0f} |")
    lat, ans = res["latency"], res["answers"]
    lines += ["", "## End-to-end latency", "",
              f"p50 {lat['total_p50_ms']:.0f} ms, p95 {lat['total_p95_ms']:.0f} ms, mean {lat['total_mean_ms']:.0f} ms "
              f"over {lat['n']} questions; {lat['within_5s_fraction'] * 100:.0f}% finished within 5 s.", "",
              "## Answer quality", ""]
    lines += [f"- **{k}**: {v:.3f}" if isinstance(v, float) else f"- **{k}**: {v}" for k, v in ans.items()]
    if "calibration" in res:
        lines += ["", "## Abstention calibration", "", "```json", json.dumps(res["calibration"], indent=2), "```"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", type=Path, default=EVAL_DIR / "eval_set.jsonl")
    ap.add_argument("--ooc", type=Path, default=EVAL_DIR / "ooc_questions.txt")
    ap.add_argument("--n-answers", type=int, default=50, help="how many questions get a full LLM answer")
    ap.add_argument("--llm-judge", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    args = ap.parse_args(argv)

    if not args.eval_set.exists():
        print(f"{args.eval_set} not found. Run: python -m scripts.build_eval_set")
        return 1
    pipe = RAGPipeline()
    if not pipe.store.count():
        print("The database is empty. Run: python -m scripts.ingest")
        return 1
    rows = load_jsonl(args.eval_set)
    ooc = [ln.strip() for ln in args.ooc.read_text().splitlines() if ln.strip()] if args.ooc.exists() else []
    print(f"Warm-up {pipe.warmup()} | {len(rows)} questions, {len(ooc)} out-of-corpus | LLM: {pipe.llm.label}")

    res = run_evaluation(pipe, rows, ooc, n_answers=args.n_answers, judge=args.llm_judge,
                         calibrate=args.calibrate, sleep=args.sleep)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(res, indent=2))
    (args.out_dir / "results.md").write_text(to_markdown(res))
    print(to_markdown(res))
    print(f"Saved {args.out_dir / 'results.json'} and results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
