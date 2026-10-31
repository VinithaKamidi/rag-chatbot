import pytest

import config
from rag.chunking import chunk_pdf
from rag.ingest import Page, ParsedPDF, extract_pdf, remove_headers_footers, tesseract_available
from tests.pdfgen import make_pdf, make_scanned_pdf


def parsed_from_texts(texts):
    pages = [Page(number=i + 1, raw="", bbox=(0, 0, 10, 10), text=t, language="en") for i, t in enumerate(texts)]
    return ParsedPDF("abc123def456", "x.pdf", pages, "en", {})


# ----------------------------------------------------------------- chunking --
def test_chunking_is_deterministic_and_covers_everything():
    texts = [" ".join(f"w{p * 250 + i}" for i in range(250)) for p in range(4)]  # 1000 words, 4 pages
    a = chunk_pdf(parsed_from_texts(texts), chunk_words=100, overlap=0.2, min_words=20)
    b = chunk_pdf(parsed_from_texts(texts), chunk_words=100, overlap=0.2, min_words=20)
    assert [(c.chunk_id, c.text) for c in a] == [(c.chunk_id, c.text) for c in b]
    assert a[0].text.split()[0] == "w0"
    assert a[-1].text.split()[-1] == "w999"
    assert a[0].page_start == 1 and a[-1].page_end == 4
    assert all(c.n_words >= 20 for c in a)


def test_overlap_is_about_twenty_percent():
    texts = [" ".join(f"w{i}" for i in range(600))]
    chunks = chunk_pdf(parsed_from_texts(texts), chunk_words=100, overlap=0.2, min_words=20)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    shared = len(set(first) & set(second))
    assert 18 <= shared <= 22


def test_chunks_prefer_sentence_boundaries():
    sentence = "This sentence has exactly eight little words here."  # 8 words
    text = " ".join([sentence] * 40)  # 320 words
    chunks = chunk_pdf(parsed_from_texts([text]), chunk_words=100, overlap=0.2, min_words=20)
    assert all(c.text.endswith(".") for c in chunks)


def test_chunk_spanning_pages_reports_range():
    chunks = chunk_pdf(parsed_from_texts(["a " * 60, "b " * 60]), chunk_words=100, overlap=0.1, min_words=10)
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert chunks[0].pages == "1–2"


def test_tiny_document_still_produces_a_chunk():
    chunks = chunk_pdf(parsed_from_texts(["just a few words"]), chunk_words=100, overlap=0.2, min_words=40)
    assert len(chunks) == 1


# ---------------------------------------------------------------- cleaning ---
def _body_line(n: int, i: int) -> str:
    """A realistic body line: lexically distinct, so it never resembles a header."""
    import random

    rng = random.Random(n * 1000 + i)
    words = "river stone market engine garden signal copper velvet harbor lantern orchid summit ember cobalt meadow".split()
    return " ".join(rng.choice(words) for _ in range(9)) + "."


def test_headers_footers_and_page_numbers_removed_but_body_kept():
    pages = []
    for n in range(1, 11):
        body = [_body_line(n, i) for i in range(30)]
        lines = ["ACME Annual Report 2024", *body, f"Page {n} of 10"]
        pages.append(Page(number=n, raw="", bbox=(0, 0, 1, 1), lines=lines))
    removed = remove_headers_footers(pages)
    for p in pages:
        assert "ACME Annual Report 2024" not in p.lines
        assert not any(ln.startswith("Page ") for ln in p.lines)
        assert len(p.lines) == 30  # every body line survived, including those at the page edges
    assert removed == 20


def test_long_edge_lines_are_never_treated_as_headers():
    long_line = "This is a long running sentence of real body content that happens to differ only by number {n} " * 2
    pages = [
        Page(number=n, raw="", bbox=(0, 0, 1, 1), lines=[long_line.format(n=n)] + [_body_line(n, i) for i in range(10)])
        for n in range(1, 9)
    ]
    assert remove_headers_footers(pages) == 0


def test_extract_native_pdf(tmp_path):
    path = make_pdf(tmp_path / "doc.pdf", "volcano", n_pages=8)
    parsed = extract_pdf(path)
    assert len(parsed.pages) == 8
    assert parsed.stats["ocr_pages"] == 0
    assert parsed.language == "en"
    joined = " ".join(p.text for p in parsed.pages)
    assert "ACME Handbook" not in joined
    assert "Page 3" not in joined
    assert "magma" in joined.lower()


# --------------------------------------------------------------------- OCR ---
@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
def test_scanned_pages_are_ocred(tmp_path):
    text = "Magma rises through the volcanic vent and erupts as lava. Seismic tremors often precede an eruption."
    path = make_scanned_pdf(tmp_path / "scan.pdf", text, n_pages=2)
    parsed = extract_pdf(path)
    assert parsed.stats["ocr_pages"] == 2
    body = " ".join(p.text for p in parsed.pages).lower()
    assert "magma" in body and "eruption" in body
    assert all(p.ocr for p in parsed.pages)


def test_ocr_can_be_disabled(tmp_path):
    path = make_scanned_pdf(tmp_path / "scan.pdf", "Some text on a scanned page for testing.", n_pages=1)
    parsed = extract_pdf(path, ocr=False)
    assert parsed.stats["ocr_pages"] == 0
    assert parsed.stats["empty_pages"] == 1
