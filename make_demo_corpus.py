"""Generate a *synthetic* test corpus: 10 PDFs >= 200 pages (built from Python's stdlib documentation, with a running
header/footer so header-stripping is exercised) + 1 scanned (image-only) PDF to exercise OCR.

For the real demo use scripts/download_corpus.py (public textbooks) or your own PDFs.
    python scripts/make_demo_corpus.py data/pdfs
"""
import io
import pkgutil
import pydoc
import sys
from pathlib import Path

import pymupdf
from fpdf import FPDF

GROUPS = {
    "python_text_and_strings": ["string", "re", "textwrap", "difflib", "unicodedata", "codecs", "json", "csv", "html", "email", "mailbox", "xml", "configparser", "shlex", "fnmatch", "glob"],
    "python_data_structures": ["collections", "heapq", "bisect", "array", "queue", "copy", "pprint", "enum", "dataclasses", "typing", "weakref", "types", "abc", "itertools", "functools", "operator"],
    "python_numeric_math": ["math", "cmath", "decimal", "fractions", "random", "statistics", "numbers", "secrets", "hashlib", "hmac", "zlib", "gzip", "bz2", "lzma", "zipfile", "tarfile"],
    "python_files_and_os": ["os", "pathlib", "shutil", "tempfile", "io", "stat", "filecmp", "fileinput", "linecache", "sqlite3", "pickle", "shelve", "dbm", "marshal", "tokenize", "zipimport"],
    "python_concurrency": ["threading", "multiprocessing", "concurrent.futures", "subprocess", "sched", "asyncio", "selectors", "signal", "queue", "contextvars", "mmap", "ctypes", "select", "socket", "ssl", "socketserver"],
    "python_networking_web": ["http.client", "http.server", "urllib.request", "urllib.parse", "urllib.error", "ftplib", "smtplib", "poplib", "imaplib", "xmlrpc.client", "xmlrpc.server", "ipaddress", "webbrowser", "wsgiref.simple_server", "cgi", "uuid"],
    "python_language_tooling": ["ast", "dis", "inspect", "importlib", "pkgutil", "trace", "traceback", "pdb", "timeit", "cProfile", "profile", "unittest", "doctest", "warnings", "logging", "argparse"],
    "python_system_runtime": ["sys", "gc", "platform", "locale", "gettext", "time", "datetime", "calendar", "zoneinfo", "getpass", "curses", "readline", "cmd", "code", "codeop", "atexit"],
    "python_gui_and_media": ["tkinter", "turtle", "wave", "colorsys", "sndhdr", "imghdr", "audioop", "aifc", "sunau", "chunk", "tkinter.ttk", "tkinter.scrolledtext", "idlelib", "pydoc", "site", "sysconfig"],
    "python_testing_and_mock": ["unittest.mock", "unittest", "doctest", "test.support", "contextlib", "tracemalloc", "faulthandler", "bdb", "pstats", "pyclbr", "py_compile", "compileall", "symtable", "tabnanny", "keyword", "token"],
}


def module_texts(names):
    for n in names:
        try:
            mod = __import__(n, fromlist=["x"])
            yield n, pydoc.plain(pydoc.render_doc(mod, renderer=pydoc.plaintext))
        except Exception:
            continue


class Doc(FPDF):
    def __init__(self, title):
        super().__init__()
        self.title_ = title

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, f"Synthetic Test Corpus - {self.title_.replace('_', ' ').title()}", align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def build(title, names, out, target_pages=210, extra=()):
    pdf = Doc(title)
    pdf.set_auto_page_break(True, 16)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, title.replace("_", " ").title(), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 8)
    texts = list(module_texts(list(names) + list(extra)))
    # cycle through the modules (re-rendering with a chapter number) until the page target is reached
    i = 0
    while pdf.page_no() < target_pages and texts:
        n, t = texts[i % len(texts)]
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, f"Chapter {i + 1}: module {n}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 8)
        for line in t.splitlines():
            pdf.multi_cell(0, 3.8, line.encode("latin-1", "replace").decode("latin-1")[:110] or " ", new_x="LMARGIN", new_y="NEXT")
            if pdf.page_no() >= target_pages:
                break
        i += 1
    pdf.output(str(out))


def scanned_pdf(out):
    """Image-only PDF (no text layer) - forces the OCR path."""
    src = pymupdf.open()
    for k in range(4):
        pg = src.new_page()
        pg.insert_textbox(pymupdf.Rect(50, 60, 545, 780),
                          f"Scanned memo {k + 1}. The quarterly maintenance window for the Zephyr-9 cooling pump is scheduled "
                          "for the third Tuesday of March. Technicians must replace the Kestrel gasket before restart.",
                          fontsize=16)
    dst = pymupdf.open()
    for pg in src:
        png = pg.get_pixmap(dpi=150).tobytes("png")
        np_ = dst.new_page()
        np_.insert_image(np_.rect, stream=png)
    dst.save(out)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pdfs")
    out.mkdir(parents=True, exist_ok=True)
    stdlib = sorted({m.name for m in pkgutil.iter_modules() if not m.name.startswith("_") and "test" not in m.name})
    for g, mods in GROUPS.items():
        print("building", g)
        build(g, mods, out / f"{g}.pdf", extra=stdlib[:60])
    scanned_pdf(out / "scanned_maintenance_memo.pdf")
    print("done")
