"""Builds small synthetic PDFs for tests (no external files needed)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import pymupdf

VOCAB = {
    "volcano": (
        "magma lava crater eruption ash plume tectonic plate vent caldera basalt sulfur "
        "pyroclastic seismic dome fissure geothermal obsidian tephra"
    ).split(),
    "bees": (
        "hive nectar pollen queen drone worker colony honeycomb wax swarm foraging "
        "pollination apiary larvae waggle dance brood propolis"
    ).split(),
}
GLUE = "the of and with near during often because while after under across toward within".split()


def filler(topic: str, n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    vocab = VOCAB[topic]
    words, out = 0, []
    while words < n_words:
        k = rng.randint(8, 14)
        sent = [rng.choice(vocab if i % 2 == 0 else GLUE) for i in range(k)]
        out.append((" ".join(sent)).capitalize() + ".")
        words += k
    return " ".join(out)


def make_pdf(
    path: Path,
    topic: str,
    n_pages: int = 12,
    facts: Optional[dict[int, str]] = None,
    header: Optional[str] = "ACME Handbook of Natural Phenomena",
    words_per_page: int = 230,
) -> Path:
    """Native-text PDF. `facts` maps 1-based page -> a sentence to plant on that page."""
    facts = facts or {}
    doc = pymupdf.open()
    for n in range(1, n_pages + 1):
        page = doc.new_page()
        if header:
            page.insert_textbox(pymupdf.Rect(50, 20, 545, 45), header, fontsize=9)
        body = filler(topic, words_per_page, seed=n * 7 + len(topic))
        if n in facts:
            body = facts[n] + " " + body
        page.insert_textbox(pymupdf.Rect(50, 70, 545, 780), body, fontsize=10)
        page.insert_textbox(pymupdf.Rect(250, 800, 345, 825), f"Page {n}", fontsize=9)
    doc.save(path)
    doc.close()
    return path


def make_scanned_pdf(path: Path, text: str, n_pages: int = 2) -> Path:
    """Image-only PDF: text is rasterised, so no native text layer exists."""
    src = pymupdf.open()
    for _ in range(n_pages):
        page = src.new_page()
        page.insert_textbox(pymupdf.Rect(60, 80, 535, 700), text, fontsize=16)
    out = pymupdf.open()
    for page in src:
        png = page.get_pixmap(dpi=150).tobytes("png")
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=png)
    out.save(path)
    out.close()
    src.close()
    return path
