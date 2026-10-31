"""Build an evaluation set from the ingested corpus.

For a seeded, per-document-balanced sample of chunks, write one question whose
answer is contained in that chunk. The gold label is the chunk's document and
page range.

    python -m scripts.build_eval_set --n 100            # questions written by the configured LLM
    python -m scripts.build_eval_set --n 100 --keyword  # no LLM: keyword queries (biased towards BM25!)

LLM-written questions tend to reuse the passage's vocabulary, which flatters
retrieval metrics. For a fairer number, add 20-30 hand-written questions to the
output file (see DOCUMENTATION.md for the format).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import config
from rag.metrics import tokenize
from rag.pipeline import RAGPipeline

DEFAULT_OUT = config.ROOT / "eval" / "eval_set.jsonl"

QUESTION_PROMPT = (
    "Write ONE specific question that can be answered using only the passage below.\n"
    "- The question must make sense on its own, without the passage.\n"
    "- Do not say \"the passage\", \"the text\" or \"the document\".\n"
    "- Do not copy long phrases verbatim.\n"
    "Output only the question.\n\nPassage:\n{passage}"
)


def eligible(row: dict, min_words: int = 150) -> bool:
    text = row["text"]
    if row["meta"].get("n_words", 0) < min_words or not text:
        return False
    # skip tables, equations and OCR noise: mostly letters and spaces only
    return sum(c.isalpha() or c.isspace() for c in text) / len(text) >= 0.85


def sample_chunks(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if eligible(r):
            by_doc[r["meta"]["pdf_id"]].append(r)
    for docs in by_doc.values():
        docs.sort(key=lambda r: r["chunk_id"])  # deterministic before shuffling
        rng.shuffle(docs)
    picked: list[dict] = []
    while len(picked) < n and any(by_doc.values()):  # round-robin across documents
        for pdf_id in sorted(by_doc):
            if by_doc[pdf_id] and len(picked) < n:
                picked.append(by_doc[pdf_id].pop())
    return picked


def keyword_question(row: dict, df: Counter, n_chunks: int, k: int = 5) -> str:
    tf = Counter(t for t in tokenize(row["text"]) if t.isalpha() and len(t) >= 4)
    scored = sorted(tf, key=lambda t: (-(tf[t] * math.log(n_chunks / df[t])), t))
    return " ".join(scored[:k])


def llm_question(pipe: RAGPipeline, row: dict) -> Optional[str]:
    try:
        out = pipe.llm.complete([{"role": "user", "content": QUESTION_PROMPT.format(passage=row["text"][:2500])}])
    except Exception as exc:  # noqa: BLE001 - rate limits etc.: skip this chunk
        print(f"  LLM error, skipping chunk: {exc}", file=sys.stderr)
        return None
    q = out.strip().strip('"').splitlines()[0].strip() if out.strip() else ""
    return q if 15 <= len(q) <= 250 and q.endswith("?") else None


def build(pipe: RAGPipeline, n: int, seed: int, keyword: bool) -> list[dict]:
    rows = list(pipe.store.iter_chunks())
    df = Counter(t for r in rows for t in set(tokenize(r["text"])))
    out = []
    for r in sample_chunks(rows, n, seed):
        q = keyword_question(r, df, len(rows)) if keyword else llm_question(pipe, r)
        if not q:
            continue
        m = r["meta"]
        out.append({
            "id": f"q{len(out) + 1:04d}", "question": q, "kind": "keyword" if keyword else "llm",
            "gold_chunk_id": r["chunk_id"], "gold_filename": m["filename"],
            "gold_page_start": m["page_start"], "gold_page_end": m["page_end"],
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--keyword", action="store_true", help="keyword queries instead of LLM-written questions")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    pipe = RAGPipeline()
    if not pipe.store.count():
        print("The database is empty. Run: python -m scripts.ingest")
        return 1
    if not args.keyword and not pipe.llm.available:
        print("No LLM configured. Set GROQ_API_KEY (or start Ollama), or pass --keyword "
              "(keyword queries favour BM25, so treat the numbers with care).")
        return 1

    rows = build(pipe, args.n, args.seed, args.keyword)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"Wrote {len(rows)} questions to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
