"""Ask a question from the terminal (handy for smoke tests and latency checks).

    python -m scripts.ask "What does the report say about X?"
"""
from __future__ import annotations

import argparse
import sys

from rag.pipeline import RAGPipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--dense-only", action="store_true")
    args = ap.parse_args(argv)

    pipe = RAGPipeline()
    if not pipe.store.count():
        print("The database is empty. Run: python -m scripts.ingest")
        return 1
    print(f"Loading models... {pipe.warmup()}  | LLM: {pipe.llm.label}\n")

    for kind, payload in pipe.ask_stream(args.question, rerank=not args.no_rerank, hybrid=not args.dense_only):
        if kind == "token":
            print(payload, end="", flush=True)
        elif kind == "done":
            ans = payload
    print("\n\nSources:")
    for s in ans.sources:
        print(f"  [{s['n']}]{'*' if s['cited'] else ' '} {s['filename']}  p.{s['pages']}")
    t = ans.timings
    print("\nLatency (ms): " + ", ".join(f"{k.replace('_ms', '')}={v:.0f}" for k, v in t.items()))
    if ans.note:
        print("Note:", ans.note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
