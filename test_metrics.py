import math

from rag.metrics import groundedness, mrr, pages_overlap, percentile, recall_at_k, strip_citations


def test_recall_and_mrr():
    ranks = [1, 3, None, 2, 10]
    assert recall_at_k(ranks, 1) == 0.2
    assert recall_at_k(ranks, 3) == 0.6
    assert recall_at_k(ranks, 10) == 0.8
    assert math.isclose(mrr(ranks), (1 + 1 / 3 + 0 + 1 / 2 + 1 / 10) / 5)
    assert recall_at_k([], 5) == 0.0 and mrr([]) == 0.0


def test_percentile_interpolates():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 50) == 3.0
    assert percentile(vals, 95) == 4.8
    assert percentile([7.0], 95) == 7.0
    assert math.isnan(percentile([], 50))


def test_pages_overlap():
    assert pages_overlap(3, 5, 5, 7) and pages_overlap(4, 4, 1, 9)
    assert not pages_overlap(1, 2, 3, 4)


def test_groundedness_flags_invented_sentences():
    context = ["The mitochondria is the organelle that produces most of the cell's ATP through respiration."]
    good = "Mitochondria produces most of the cell's ATP through respiration [1]."
    bad = good + " Scientists discovered that mitochondria were originally shipped from Jupiter by ancient astronauts."
    assert groundedness(good, context)[0] == 1.0
    frac, unsupported = groundedness(bad, context)
    assert frac == 0.5 and "Jupiter" in unsupported[0]


def test_strip_citations():
    assert strip_citations("Fact [1][2, 3] here [4].").split() == ["Fact", "here", "."]
    assert strip_citations("keep array[0] intact") == "keep array[0] intact"
