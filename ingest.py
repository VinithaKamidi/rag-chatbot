"""Ingest PDFs into the vector database.

    python -m scripts.ingest                      # everything in data/pdfs
    python -m scripts.ingest path/to/a.pdf dir/   # specific files or folders
    python -m scripts.ingest --force --no-ocr
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import config
from rag.pipeline import RAGPipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="PDF files or folders (default: data/pdfs)")
    ap.add_argument("--force", action="store_true", help="re-ingest even if already in the database")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR for scanned pages")
    args = ap.parse_args(argv)

    targets = args.paths or [config.PDF_DIR]
    pdfs: list[Path] = []
    for t in targets:
        pdfs += sorted(t.glob("*.pdf")) if t.is_dir() else [t]
    if not pdfs:
        print(f"No PDFs found in {', '.join(map(str, targets))}. Put your PDFs in {config.PDF_DIR}.")
        return 1

    pipe = RAGPipeline()
    t_all = time.perf_counter()
    last = {"stage": None}

    def progress(stage, done, total):
        if stage != last["stage"] or done == total:
            print(f"    {stage:<8} {done}/{total}", end="\r" if done < total else "\n", flush=True)
            last["stage"] = stage

    rows = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        res = pipe.ingest_pdf(pdf, force=args.force, ocr=False if args.no_ocr else None, progress=progress)
        rows.append(res)
        if res["status"] == "skipped":
            print("    already ingested (use --force to redo)")
        else:
            s = res["seconds"]
            print(f"    {res['pages']} pages ({res['ocr_pages']} OCR) -> {res['chunks']} chunks | "
                  f"extract+ocr {s['extract_ocr']}s, embed {s['embed']}s, index {s['index']}s")

    done = [r for r in rows if r["status"] == "ingested"]
    print(f"\nIngested {len(done)} new PDF(s), skipped {len(rows) - len(done)}. "
          f"Database now holds {pipe.store.count()} chunks from {len(pipe.store.documents())} PDFs "
          f"({time.perf_counter() - t_all:.0f}s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
