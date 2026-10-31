FROM python:3.13-slim

# Tesseract is the OCR engine used for scanned pages and image-only PDFs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (Hugging Face Spaces requires uid 1000; good practice anywhere).
RUN useradd -m -u 1000 app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PORT=7860

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model and reranker into the image so start-up never
# depends on the model hub being reachable. (If you change EMBED_MODEL or
# RERANK_MODEL, change them here too.)
RUN python -c "from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY . .
RUN chown -R app:app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT','7860'))"

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
