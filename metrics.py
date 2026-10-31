"""Evaluation metrics. Pure functions, no model or database access."""
from __future__ import annotations

import math
import re
from typing import Optional, Sequence

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "which", "with",
    "can", "will", "also", "such", "than", "then", "these", "those", "into", "not", "but", "if",
    "they", "he", "she", "we", "you", "i", "do", "does", "did", "how", "what", "when", "where", "who",
    "why", "about", "between", "during", "other", "more", "most", "some", "any", "each", "may",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


def recall_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """`ranks` holds the 1-based rank of the first relevant result, or None."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[Optional[int]]) -> float:
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100)."""
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def pages_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text) if s.strip()]


_CITE = re.compile(r"\[[1-9]\d?(?:\s*,\s*[1-9]\d?)*\]")


def strip_citations(text: str) -> str:
    return _CITE.sub("", text)


def groundedness(answer: str, context: Sequence[str], threshold: float = 0.5, min_tokens: int = 4):
    """Lexical groundedness proxy.

    A sentence is *supported* when at least `threshold` of its content words
    occur in the retrieved context. Returns (fraction_supported, unsupported).
    This is a cheap, deterministic proxy. It misses subtle contradictions and
    can penalise heavy paraphrase, so treat it as a screen, not as ground truth.
    """
    ctx = set(tokenize(" ".join(context)))
    checked, unsupported = 0, []
    for sent in sentences(strip_citations(answer)):
        toks = tokenize(sent)
        if len(toks) < min_tokens:
            continue
        checked += 1
        if sum(t in ctx for t in toks) / len(toks) < threshold:
            unsupported.append(sent)
    if checked == 0:
        return 1.0, []
    return 1.0 - len(unsupported) / checked, unsupported
