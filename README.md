---
title: PDF RAG Chatbot
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PDF RAG Chatbot (open-source stack)

Ask questions over a private library of large PDFs (10+ PDFs, 200+ pages each, native text **and** scanned pages) and get
answers with **PDF name + page citations** in a few seconds.

| Stage | Tool (all free / open source) |
|---|---|
| PDF text extraction | PyMuPDF |
| OCR for scanned pages | Tesseract (via pytesseract) |
| Embeddings | `BAAI/bge-small-en-v1.5` (ONNX, CPU) via fastembed |
| Vector DB + ANN index | ChromaDB, HNSW (cosine) |
| Keyword search | BM25 (rank-bm25) |
| Reranker | cross-encoder `ms-marco-MiniLM-L-6-v2` (ONNX) |
| LLM | Llama-3.1-8B on Groq's free tier, **or** local Ollama, **or** extractive fallback (no LLM) |
| UI | Streamlit: chat, retrieval visualisation, ingestion pipeline, evaluation dashboard |

Full design, trade-offs, evaluation method and deployment: **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Quick start

```bash
# 1. system dependency for OCR
sudo apt-get install tesseract-ocr            # macOS: brew install tesseract

# 2. python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. (optional but recommended) free LLM key from console.groq.com -> .env
echo "GROQ_API_KEY=gsk_..." > .env

# 4. get 10+ big PDFs (or copy your own into data/pdfs/)
python -m scripts.download_corpus
python -m scripts.check_corpus                # verifies >= 10 PDFs, each >= 200 pages

# 5. ingest: extract/OCR -> clean -> chunk -> embed -> HNSW index   (one-off, resumable)
python -m scripts.ingest

# 6. chat
streamlit run app.py
```

Terminal check: `python -m scripts.ask "your question"` prints the answer, sources and a latency breakdown.

## Evaluate

```bash
python -m scripts.build_eval_set --n 100      # question per sampled chunk (LLM-written; --keyword if no LLM)
python -m scripts.evaluate                    # Recall@k, MRR, p50/p95 latency, citation validity, hallucination proxy
python -m scripts.evaluate --calibrate        # suggests abstention thresholds for your corpus
python -m pytest                              # 39 unit/integration tests (no network needed)
```

Results land in `eval/results.json` / `eval/results.md` and appear in the app's **Evaluation** tab.

## Deploy

See *Deployment* in [DOCUMENTATION.md](DOCUMENTATION.md): Streamlit Community Cloud straight from this GitHub repo
(`packages.txt` installs Tesseract), or Hugging Face Spaces / any host via the included `Dockerfile`.
