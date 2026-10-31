"""Download public, freely available textbooks (each >= 200 pages) as the demo corpus.

    python -m scripts.download_corpus            # -> data/pdfs
    python -m scripts.check_corpus               # verify: >= 10 PDFs, each >= 200 pages

URLs are public and were NOT verified from the build sandbox (no internet there): the script skips anything that
fails or is shorter than 200 pages and tells you how many valid PDFs you have. Add your own PDFs if you are short.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pymupdf

import config

URLS = {
    "elements_of_statistical_learning.pdf": "https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf",
    "islr2_statistical_learning.pdf": "https://hastie.su.domains/ISLR2/ISLRv2_corrected_June_2023.pdf",
    "dive_into_deep_learning.pdf": "https://d2l.ai/d2l-en.pdf",
    "think_python_2e.pdf": "https://greenteapress.com/thinkpython2/thinkpython2.pdf",
    "intro_to_information_retrieval.pdf": "https://nlp.stanford.edu/IR-book/pdf/irbookprint.pdf",
    "speech_and_language_processing_3e.pdf": "https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf",
    "information_theory_inference_learning.pdf": "https://www.inference.org.uk/itprnn/book.pdf",
    "mathematics_for_machine_learning.pdf": "https://mml-book.github.io/book/mml-book.pdf",
    "think_bayes.pdf": "https://greenteapress.com/thinkbayes/thinkbayes.pdf",
    "think_stats_2e.pdf": "https://greenteapress.com/thinkstats2/thinkstats2.pdf",
    "gnu_scientific_library_manual.pdf": "https://www.gnu.org/software/gsl/doc/latex/gsl-ref.pdf",
    "pro_git.pdf": "https://github.com/progit/progit2/releases/download/2.1.434/progit.pdf",
}


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else config.PDF_DIR
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name, url in URLS.items():
        dest = out / name
        try:
            if not dest.exists():
                with httpx.stream("GET", url, follow_redirects=True, timeout=180) as r:
                    r.raise_for_status()
                    dest.write_bytes(b"".join(r.iter_bytes()))
            n = len(pymupdf.open(dest))
            if n < 200:
                print(f"SKIP {name}: only {n} pages"); dest.unlink(); continue
            print(f"OK   {name}: {n} pages"); ok += 1
        except Exception as exc:
            print(f"FAIL {name}: {exc}"); dest.unlink(missing_ok=True)
    print(f"\n{ok} valid PDFs in {out}" + ("" if ok >= 10 else "  <-- need 10: add more PDFs of your own"))
    return 0 if ok >= 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
