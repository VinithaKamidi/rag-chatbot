# Documentation: RAG Chatbot over Large PDFs

## 1. Problem and how this solution meets it

**Brief:** a chatbot over ≥10 PDFs of ≥200 pages each (text, images, scans) using free/open-source embeddings and vector DB,
answering in 2–5 s with PDF-name + page citations, and a demo that shows ingestion, retrieved chunks and the cited answer.

| Requirement | How it is met | Where |
|---|---|---|
| Native text + OCR | PyMuPDF text layer; any page with < 40 characters is rendered at 200 DPI and OCR'd with Tesseract (thread pool) | `rag/ingest.py` |
| Clean / normalise / headers / language | Unicode NFKC, de-hyphenation, repeated header/footer line removal, per-page `langdetect` (seeded) | `rag/ingest.py` |
| Chunking 500–1000 tok, 10–30 % overlap | 375-word windows (~500 tokens), 20 % overlap, sentence-boundary aware, deterministic ids | `rag/chunking.py` |
| Metadata | pdf id, filename, chunk index, page start/end, language, text-region bounding box | `rag/chunking.py` |
| Open-source embeddings | BGE-small-en-v1.5, 384-d, computed once and persisted | `rag/embeddings.py` |
| Open-source vector DB + ANN | ChromaDB persistent, HNSW cosine index | `rag/store.py` |
| Top-K with metadata and scores | dense + BM25 hybrid, every hit carries dense score/rank, BM25 score/rank, fused and rerank scores | `rag/retrieval.py` |
| Lightweight reranker | cross-encoder MiniLM-L6 over 15 fused candidates (lexical fallback if the model is unavailable) | `rag/retrieval.py` |
| LLM with provenance | numbered sources in the prompt, the model must cite `[n]`; sources list shows `file · pages` | `rag/generation.py` |
| 2–5 s latency | streaming answer, small models, tiny prompt (5 chunks), measured p50/p95 in the app and eval | §6 |
| Evaluation & monitoring | Recall@k, MRR, p95 latency, citation validity/hit, hallucination proxy, out-of-corpus abstention | `scripts/evaluate.py` |
| Reproducibility | deterministic chunking, content-hash PDF ids, idempotent ingestion, pinned dependencies | §4 |
| Demo UI | Streamlit tabs: Chat, Ingestion pipeline, Evaluation | `app.py` |

## 2. Architecture

```
            INGESTION (offline, once per PDF)                          QUERY (online)
 ┌─────┐   ┌──────────────┐   ┌───────┐   ┌────────┐   ┌─────────┐        question
 │ PDF │──▶│ extract text │──▶│ clean │──▶│ chunk  │──▶│  embed  │──┐        │
 └─────┘   │ (OCR if scan)│   │ +lang │   │+meta   │   │  (BGE)  │  │        ▼
           └──────────────┘   └───────┘   └────────┘   └─────────┘  │   embed query (BGE)
                                                                     ▼        │
                                                          ┌────────────────┐  ├─▶ dense ANN (HNSW, top 30)
                                                          │  ChromaDB      │◀─┤
                                                          │  HNSW + meta   │  └─▶ BM25 (top 30)
                                                          └────────────────┘        │
                                                                                    ▼
                                                              Reciprocal Rank Fusion → top 15
                                                                                    ▼
                                                              cross-encoder rerank → top 5
                                                                                    ▼
                                                              abstain? ── yes ─▶ "couldn't find this"
                                                                  │ no
                                                                  ▼
                                                  LLM (Groq/Ollama) or extractive answer, with [n] citations
                                                                  ▼
                                                  validate citations → answer + sources + latency breakdown
```

## 3. Design decisions (and why)

**Why hybrid retrieval (dense + BM25 + RRF) rather than dense only?** Dense embeddings capture meaning but blur exact
identifiers (error codes, part numbers, names), which are common in large technical PDFs. BM25 catches those. Reciprocal Rank
Fusion merges the two rank lists without needing comparable score scales. `scripts/evaluate.py` reports dense / hybrid /
hybrid+rerank side by side so the benefit is measured on your corpus, not assumed.

**Why a cross-encoder reranker on only 15 candidates?** A cross-encoder reads query and passage together and is much more precise
than a bi-encoder, but it is too slow to run on thousands of chunks. Running it on the 15 fused candidates keeps the cost small
(tens to low hundreds of ms on CPU).

**Why BGE-small?** 384 dimensions, MIT-licensed, strong retrieval quality for its size, runs on CPU through ONNX (no PyTorch),
so cold starts and per-query embedding are fast. 512-token input limit is why chunks are ~375 words: larger chunks would be
silently truncated by the model.

**Why ChromaDB?** Embedded (no server), persistent, open source, real HNSW index and metadata filtering. Parameters are in
`rag/store.py`; `search_ef` trades recall for latency.

**Why abstention?** A chatbot that confidently answers out-of-scope questions is worse than one that says "not found". Two guards:
a minimum dense similarity (`MIN_DENSE_SCORE`) and an optional cross-encoder score floor (`MIN_RERANK_SCORE`). Both are
model-specific; `python -m scripts.evaluate --calibrate` computes suggested values from in-corpus vs out-of-corpus questions
(`eval/ooc_questions.txt`).

**Why numbered citations (`[1]`) instead of asking the LLM to write file names?** Small models mis-copy filenames and page numbers.
Numbers are easy for them; the application maps `[n]` back to the real filename and pages, and checks every `[n]` points at a
source that was actually provided (`citation_validity`).

**Why an extractive fallback?** If no LLM key/Ollama is available, or the LLM call fails or times out, the app returns the best
matching sentences with citations. The demo never dies, and the retrieval half of the system can be judged on its own.

## 4. Ingestion details

1. **Extract**: `page.get_text()` per page. Pages with fewer than `OCR_MIN_CHARS` (40) characters are treated as scanned.
2. **OCR**: those pages are rendered at `OCR_DPI` (200) and run through Tesseract in a thread pool (`OCR_WORKERS`). Rendering stays
   on one thread because PyMuPDF documents are not thread-safe. Pages are processed in bounded batches so memory stays flat on
   large scans. Other languages: install the Tesseract language pack and set `OCR_LANG=eng+hin` etc.
3. **Clean**: Unicode NFKC (ligatures, non-breaking spaces), remove hyphenation at line breaks, collapse whitespace. Lines in the top/bottom
   3 lines of a page that repeat (digits ignored) on ≥ 40 % of pages are dropped, which removes running headers, footers and page numbers
   without touching body text. A language is detected per page (seeded, deterministic).
4. **Chunk**: pages are flattened into one word stream that remembers each word's page; windows of 375 words with 20 % overlap, preferring
   to end at a sentence boundary in the last 15 % of the window. Chunk id = `{pdf_id}-{index}`; `pdf_id` is a hash of the file bytes.
5. **Embed** in batches of 64 and **write** to ChromaDB with metadata.
6. **Idempotent**: re-running ingestion skips PDFs already in the manifest (`--force` re-indexes). Renaming a file does not duplicate it.

**Bounding boxes:** each chunk stores the text region of its first page (`x0,y0,x1,y1`, PDF points). Per-line boxes were not stored,
since page-level citation is what the answer exposes.

## 5. Generation

The prompt contains the question and up to 5 numbered passages, with rules to answer only from them, cite `[n]`, and reply with the fixed
sentence "I couldn't find this in the provided documents." when the sources do not contain the answer. Text inside passages is treated as
data (basic prompt-injection hygiene). Temperature is 0.1.

Providers (`LLM_PROVIDER=auto`): Groq free tier if `GROQ_API_KEY` is set (Llama-3.1-8B-instant), else a local Ollama server if running
(`llama3.2:3b`), else the extractive fallback. Any OpenAI-compatible endpoint works with `LLM_PROVIDER=openai_compat`, `LLM_BASE_URL`, `LLM_MODEL`.
Answers are streamed, so the first words appear while the rest is generated.

## 6. Latency (2–5 s target)

End-to-end time = query embedding + HNSW search + BM25 + fusion + rerank + LLM. Everything except the LLM is local and CPU-light;
the LLM is the dominant, variable term. The levers, in order of impact:

| Lever | Setting |
|---|---|
| Fast hosted or small local LLM | Groq 8B (sub-second generation) or Ollama 3B |
| Fewer / shorter passages in the prompt | `TOP_K_FINAL=5`, `LLM_MAX_TOKENS=350` |
| Skip LLM when irrelevant | abstention guards |
| Fewer rerank candidates | `RERANK_CANDIDATES` (default 15) |
| HNSW recall/speed | `search_ef` in `rag/store.py` |
| Models pre-loaded at start-up | `pipeline.warmup()` runs at app start, so the first user query is not slow |
| Models baked into the Docker image | no downloads at cold start |

The UI shows a latency breakdown for each answer (embed / dense / BM25 / rerank / time-to-first-token / total); `scripts/evaluate.py`
reports p50/p95 for full question→answer runs and retrieval only. **Measure on your hardware**: the 2–5 s figure depends on CPU and LLM provider.

## 7. Evaluation method

`python -m scripts.build_eval_set --n 100` samples chunks (seeded, balanced per PDF) and has the LLM write one question each; the gold label
is the chunk's PDF and page range. `--keyword` builds keyword queries without an LLM (biased toward BM25: read those numbers with care).
LLM-written questions reuse the passage's vocabulary and flatter retrieval, so **add 20–30 hand-written questions** to `eval/eval_set.jsonl` for a fairer number. Each line:

```json
{"id": "q0101", "question": "…?", "kind": "manual", "gold_filename": "book.pdf", "gold_page_start": 112, "gold_page_end": 113}
```

| Metric | Definition |
|---|---|
| Recall@1/3/5/10, MRR | a hit is "relevant" if it is from the gold PDF and its page range overlaps the gold pages (page-level, matching what citations show); `chunk_recall@5` is the strict exact-chunk version |
| Dense vs hybrid vs hybrid+rerank | same questions, three configurations, to show what each stage adds |
| Latency p50 / p95 | end-to-end question→answer, and retrieval-only |
| citation_validity | answers whose `[n]` markers all point at provided sources |
| citation_hit | answers that cite a source on the gold page |
| hallucination_rate_proxy | answers with a sentence not lexically supported by the retrieved text. **A screen, not ground truth**: it misses fluent but wrong paraphrases. `--llm-judge` adds an LLM-based faithfulness check |
| ooc_abstention_rate | out-of-corpus questions (`eval/ooc_questions.txt`) correctly refused |

### Results

Run on your corpus and paste `eval/results.md` here before submitting:

```bash
python -m scripts.ingest && python -m scripts.build_eval_set --n 100 && python -m scripts.evaluate
```

> No retrieval-quality or latency numbers are quoted in this document on purpose: the build environment had no access to the embedding
> model, so numbers produced there would not describe the real system. All 39 tests pass; the pipeline was exercised end to end
> (10 synthetic PDFs of 210 pages + a scanned PDF through OCR, 1,971 chunks, 28 ms retrieval with the offline test embedder) but that
> embedder is non-semantic and is not used for the demo.

## 8. Running it

See the README quick start. Configuration is by environment variables or a `.env` file; every option and default is in `config.py`
(chunk size/overlap, OCR settings, retrieval depths, model names, abstention thresholds, LLM provider). `RAG_EMBEDDER=hashing` is a
no-download test embedder used only by the offline tests; never use it for the demo.

## 9. Deployment (GitHub link + live demo)

**GitHub**
```bash
git init && git add . && git commit -m "RAG chatbot over large PDFs"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
PDFs are git-ignored (large and possibly private). To ship a ready-to-query demo, build the index locally with the real embedder
(`python -m scripts.ingest`), then commit `data/chroma/` and `data/manifest.json` (precomputed embeddings; typically tens of MB for
~5,000 chunks; use Git LFS if a file exceeds 100 MB). This is also what makes the demo reproducible.

**Option A, Streamlit Community Cloud (fastest from a GitHub link).** share.streamlit.io → New app → pick the repo, main file `app.py`.
`packages.txt` installs Tesseract; `requirements.txt` installs Python deps. Add `GROQ_API_KEY` under *Settings → Secrets*. Free instances are small and slow
on first start (model download), so open the app once before the demo.

**Option B, Hugging Face Spaces (Docker).** Create a Space with the Docker SDK, and add this front matter to the top of the Space's copy of `README.md`:
```yaml
---
title: PDF RAG Chatbot
sdk: docker
app_port: 7860
---
```
Push the repo to the Space remote. The `Dockerfile` bakes the embedding model and reranker into the image, runs as uid 1000, and serves Streamlit on port 7860.
Add `GROQ_API_KEY` as a Space secret.

## 10. Limitations and next steps

- English-centric: the embedder is English BGE-small; for multilingual corpora switch `EMBED_MODEL` to a multilingual model and install OCR language packs.
- Tables and figures inside PDFs are read as text (or OCR'd); table structure is not preserved and figure content is not described.
- The hallucination metric is a lexical proxy; a human-labelled set or an LLM judge is stronger.
- Free LLM tiers are rate-limited; for heavy use run Ollama locally or a paid endpoint.
- Single-process, single-collection design: fine for tens of large PDFs; beyond that, use a server-mode vector DB (Qdrant, Chroma server) and a job queue for ingestion.

## 11. Repository layout

```
app.py                 Streamlit UI (chat · ingestion pipeline · evaluation)
config.py              all settings, env-overridable
rag/ingest.py          extract, OCR, clean, headers/footers, language
rag/chunking.py        deterministic page-aware chunking + metadata
rag/embeddings.py      BGE via fastembed (+ offline hashing embedder for tests)
rag/store.py           ChromaDB HNSW store + manifest
rag/retrieval.py       dense + BM25 + RRF + cross-encoder rerank
rag/generation.py      prompts, LLM providers, extractive fallback, citation checks
rag/pipeline.py        orchestration (ingest_pdf / ask / ask_stream / warmup)
rag/metrics.py         Recall@k, MRR, groundedness, percentiles
scripts/               ingest · ask · download_corpus · check_corpus · build_eval_set · evaluate · make_demo_corpus
eval/                  eval sets, out-of-corpus questions, results
tests/                 39 tests
Dockerfile, packages.txt, requirements*.txt, .streamlit/
```
