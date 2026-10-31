"""Check a PDF folder against the brief: at least 10 PDFs, each at least 200 pages.

    python -m scripts.check_corpus [data/pdfs]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf

import config

MIN_PDFS, MIN_PAGES = 10, 200


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", type=Path, default=config.PDF_DIR)
    args = ap.parse_args(argv)

    pdfs = sorted(args.folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {args.folder}")
        return 1
    rows, ok = [], 0
    for p in pdfs:
        try:
            with pymupdf.open(p) as doc:
                pages = doc.page_count
                scanned = sum(1 for i in range(min(pages, 20)) if len(doc[i].get_text().strip()) < config.OCR_MIN_CHARS)
                sample = min(pages, 20)
        except Exception as exc:  # noqa: BLE001
            rows.append((p.name, "unreadable", str(exc)[:40], False))
            continue
        good = pages >= MIN_PAGES
        ok += good
        rows.append((p.name, pages, f"{scanned}/{sample} sampled pages look scanned", good))

    width = max(len(r[0]) for r in rows)
    for name, pages, note, good in rows:
        print(f"{'OK ' if good else 'LOW'}  {name:<{width}}  {str(pages):>6} pages  {note}")
    meets = len(pdfs) >= MIN_PDFS and ok >= MIN_PDFS
    print(f"\n{len(pdfs)} PDFs, {ok} with >= {MIN_PAGES} pages. Brief ({MIN_PDFS}+ PDFs of {MIN_PAGES}+ pages): "
          f"{'MET' if meets else 'NOT MET'}")
    return 0 if meets else 2


if __name__ == "__main__":
    sys.exit(main())
