"""Streamlit demo: chat with citations, retrieval visualisation, ingestion pipeline, evaluation.

    streamlit run app.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from rag.metrics import percentile, tokenize
from rag.pipeline import Answer, RAGPipeline

EVAL_RESULTS = config.ROOT / "eval" / "results.json"
STAGE_LABELS = {"extract": "1. Extract text", "ocr": "2. OCR scanned pages", "clean": "3. Clean + language",
                "embed": "4. Embed chunks", "index": "5. Write to HNSW index"}

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models and building the search index...")
def get_pipeline() -> tuple[RAGPipeline, dict]:
    pipe = RAGPipeline()
    return pipe, pipe.warmup()


pipe, warm = get_pipeline()
docs = pipe.store.documents()


# ----------------------------------------------------------------- helpers ---
def highlight(text: str, query: str) -> str:
    """HTML-escape `text` and wrap query terms in <mark> (split first, so entities are never corrupted)."""
    terms = sorted({t for t in tokenize(query)}, key=len, reverse=True)
    if not terms:
        return html.escape(text)
    parts = re.split(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", text, flags=re.I)
    return "".join(f"<mark>{html.escape(p)}</mark>" if i % 2 else html.escape(p) for i, p in enumerate(parts))


def md_safe(text: str) -> str:
    return text.replace("$", "\\$")  # keep "$" from being read as LaTeX


def render_details(ans: Answer, key: str, expanded: bool) -> None:
    t = ans.timings
    total_s = t.get("total_ms", 0) / 1000
    lo, hi = config.LATENCY_TARGET_S

    cols = st.columns(4)
    cols[0].metric("End-to-end", f"{total_s:.2f} s", "within 5 s target" if total_s <= hi else "over 5 s target",
                   delta_color="normal" if total_s <= hi else "inverse")
    cols[1].metric("Retrieval", f"{t.get('retrieval_ms', 0):.0f} ms")
    cols[2].metric("First token", f"{t['ttft_ms']:.0f} ms" if "ttft_ms" in t else "n/a")
    cols[3].metric("Generation", f"{t.get('generate_ms', 0):.0f} ms")

    st.markdown("**Sources**")
    if ans.abstained:
        st.caption("No supporting passage was found. Closest passages retrieved (not cited):")
    for s in ans.sources:
        mark = "✅ cited" if s["cited"] else "·"
        st.markdown(f"`[{s['n']}]` **{s['filename']}** — page {s['pages']} &nbsp; {mark}")
    if ans.invalid_citations:
        st.warning(f"The model cited source number(s) {ans.invalid_citations} that were not provided.")
    if ans.note:
        st.caption(f"ℹ️ {ans.note}")

    with st.expander("🔍 Retrieval visualisation: top retrieved chunks", expanded=expanded):
        rows = [{
            "#": i, "Document": h.filename, "Pages": h.pages,
            "Dense cosine": None if h.dense_score is None else round(h.dense_score, 3),
            "BM25 rank": h.bm25_rank, "Dense rank": h.dense_rank,
            "Fused (RRF)": round(h.fused_score, 4),
            "Rerank score": None if h.rerank_score is None else round(h.rerank_score, 3),
            "OCR": bool(h.meta.get("ocr")),
        } for i, h in enumerate(ans.hits, 1)]
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        score_col = "Rerank score" if df["Rerank score"].notna().any() else "Fused (RRF)"
        st.caption(f"Bar length = {score_col} (the score that decided the final order)")
        st.bar_chart(df.set_index("#")[score_col], horizontal=True, height=40 + 34 * len(df))
        for i, h in enumerate(ans.hits, 1):
            with st.expander(f"[{i}] {h.filename} — p.{h.pages}", expanded=False):
                st.markdown(highlight(h.text, ans.question), unsafe_allow_html=True)
                st.caption(f"chunk {h.chunk_id} · {h.meta.get('n_words')} words · language {h.meta.get('language')} "
                           f"· bbox {h.meta.get('bbox')}")

    with st.expander("⏱ Latency breakdown (ms)"):
        parts = {"embed query": t.get("embed_ms"), "dense ANN": t.get("dense_ms"), "BM25": t.get("bm25_ms"),
                 "rerank": t.get("rerank_ms"), "LLM generate": t.get("generate_ms")}
        st.bar_chart(pd.Series({k: round(v, 1) for k, v in parts.items() if v is not None}), horizontal=True)


# ----------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.title("📚 PDF RAG Chatbot")
    n_pages = sum(d.get("pages", 0) for d in docs.values())
    st.metric("Documents", len(docs))
    c1, c2 = st.columns(2)
    c1.metric("Pages", f"{n_pages:,}")
    c2.metric("Chunks", f"{pipe.store.count():,}")
    st.caption(f"Embeddings: `{getattr(pipe.embedder, 'name', '?')}`  \nReranker: `{pipe.reranker.name}`  \n"
               f"LLM: `{pipe.llm.label}`  \nVector DB: ChromaDB (HNSW, cosine)")
    if not pipe.llm.available:
        st.warning("No LLM configured, so answers are extractive. Set `GROQ_API_KEY` or start Ollama.")
    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Passages sent to the LLM", 2, 10, config.TOP_K_FINAL)
    use_hybrid = st.toggle("Hybrid search (dense + BM25)", value=True)
    use_rerank = st.toggle("Cross-encoder rerank", value=True)
    names = sorted({d["filename"] for d in docs.values()})
    chosen = st.multiselect("Limit to documents", names, placeholder="All documents")
    pdf_ids = [i for i, d in docs.items() if d["filename"] in chosen] or None
    st.caption(f"Start-up warm-up: {warm.get('warmup_s', 0)} s")

tab_chat, tab_ingest, tab_eval = st.tabs(["💬 Chat", "📥 Ingestion pipeline", "📊 Evaluation"])

# -------------------------------------------------------------------- chat ---
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not docs:
        st.info("The knowledge base is empty. Upload PDFs in the **Ingestion pipeline** tab, "
                "or run `python -m scripts.ingest`.")

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(md_safe(m["content"]))
            if m.get("answer"):
                render_details(m["answer"], key=f"h{i}", expanded=False)

    question = st.chat_input("Ask a question about your documents", disabled=not docs)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            holder: dict = {}

            def stream():
                for kind, payload in pipe.ask_stream(question, hybrid=use_hybrid, rerank=use_rerank,
                                                     top_k=top_k, pdf_ids=pdf_ids):
                    if kind == "token":
                        yield md_safe(payload)
                    elif kind == "done":
                        holder["answer"] = payload

            with st.spinner("Retrieving..."):
                st.write_stream(stream())
            ans: Answer = holder["answer"]
            render_details(ans, key="latest", expanded=True)
        st.session_state.messages.append({"role": "assistant", "content": ans.text, "answer": ans})

# --------------------------------------------------------------- ingestion ---
with tab_ingest:
    st.subheader("Ingestion pipeline")
    st.graphviz_chart(
        'digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontsize=11];'
        ' pdf [label="PDF files"]; ext [label="Extract text\\n(PyMuPDF)"]; ocr [label="OCR scanned pages\\n(Tesseract)"];'
        ' cln [label="Clean\\nheaders/footers, language"]; chk [label="Chunk\\n~375 words, 20% overlap\\n+ page metadata"];'
        ' emb [label="Embed\\n(bge-small, ONNX)"]; idx [label="HNSW index\\n(ChromaDB)"];'
        ' pdf -> ext -> cln -> chk -> emb -> idx; ext -> ocr [style=dashed, label="sparse text"]; ocr -> cln; }'
    )

    up = st.file_uploader("Add PDFs to the knowledge base", type="pdf", accept_multiple_files=True)
    c_force, c_ocr = st.columns(2)
    force = c_force.checkbox("Re-ingest if already present", value=False)
    do_ocr = c_ocr.checkbox("OCR scanned pages", value=config.OCR_ENABLED)
    if up and st.button("Ingest uploaded PDFs", type="primary"):
        config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        for f in up:
            dest = config.PDF_DIR / Path(f.name).name  # basename only: no path traversal
            dest.write_bytes(f.getbuffer())
            bar, label = st.progress(0.0), st.empty()
            stage_order = list(STAGE_LABELS)

            def progress(stage, done, total, _bar=bar, _label=label, _name=f.name):
                if stage in stage_order:
                    frac = (stage_order.index(stage) + (done / total if total else 1)) / len(stage_order)
                    _bar.progress(min(frac, 1.0))
                    _label.caption(f"{_name}: {STAGE_LABELS[stage]} ({done}/{total})")

            res = pipe.ingest_pdf(dest, force=force, ocr=do_ocr, progress=progress)
            bar.progress(1.0)
            if res["status"] == "skipped":
                label.info(f"{f.name}: already in the knowledge base (tick *Re-ingest* to redo it).")
            else:
                s = res["seconds"]
                label.success(f"{f.name}: {res['pages']} pages ({res['ocr_pages']} via OCR) → {res['chunks']} chunks. "
                              f"Extract/OCR {s['extract_ocr']}s · embed {s['embed']}s · index {s['index']}s")
        st.rerun()

    docs = pipe.store.documents()
    if docs:
        df = pd.DataFrame([{
            "File": d["filename"], "Pages": d.get("pages"), "OCR pages": d.get("ocr_pages"),
            "Chunks": d.get("chunks"), "Words": d.get("words"), "Lang": d.get("language"),
            "Headers/footers removed": d.get("boilerplate_lines_removed"),
            "Ingest time (s)": round(sum(d.get("seconds", {}).values()), 1),
        } for d in docs.values()]).sort_values("File")
        st.dataframe(df, hide_index=True, width="stretch")
        big = sum(1 for d in docs.values() if d.get("pages", 0) >= 200)
        st.caption(f"Brief requirement (≥10 PDFs, each ≥200 pages): {len(docs)} PDFs ingested, {big} with ≥200 pages.")

        st.subheader("Inspect a chunk")
        pick = st.selectbox("Document", sorted(docs.values(), key=lambda d: d["filename"]),
                            format_func=lambda d: d["filename"])
        idx = st.number_input("Chunk index", 0, max(pick["chunks"] - 1, 0), 0)
        row = pipe.store.get_chunk(f"{pick['pdf_id']}:{int(idx):05d}")
        if row:
            st.code(row["text"], language=None, wrap_lines=True)
            st.json(row["meta"])

# -------------------------------------------------------------- evaluation ---
with tab_eval:
    st.subheader("Offline evaluation")
    if EVAL_RESULTS.exists():
        res = json.loads(EVAL_RESULTS.read_text())
        st.caption(f"Run: {res['meta'].get('date')} · {res['meta'].get('n_questions')} questions "
                   f"· embeddings `{res['meta'].get('embed_model')}` · LLM `{res['meta'].get('llm')}`")
        st.markdown("**Retrieval** (page-level match with the source passage)")
        st.dataframe(pd.DataFrame(res["retrieval"]).T.round(3), width="stretch")
        a, b = st.columns(2)
        a.markdown("**Latency**")
        a.dataframe(pd.Series(res["latency"], name="value").round(1).to_frame(), width="stretch")
        b.markdown("**Answer quality**")
        b.dataframe(pd.Series(res["answers"], name="value").round(3).to_frame(), width="stretch")
    else:
        st.info("No results yet. Run `python -m scripts.build_eval_set` then `python -m scripts.evaluate`.")

    st.divider()
    st.subheader("Live latency probe")
    default_q = "\n".join(m["content"] for m in st.session_state.get("messages", []) if m["role"] == "user")[:600]
    qs = st.text_area("One question per line", value=default_q, height=120)
    if st.button("Run probe") and docs:
        lat = []
        prog = st.progress(0.0)
        lines = [q.strip() for q in qs.splitlines() if q.strip()]
        for i, q in enumerate(lines, 1):
            a = pipe.ask(q, hybrid=use_hybrid, rerank=use_rerank, top_k=top_k, pdf_ids=pdf_ids)
            lat.append(a.timings["total_ms"] / 1000)
            prog.progress(i / len(lines))
        if lat:
            m = st.columns(3)
            m[0].metric("p50", f"{percentile(lat, 50):.2f} s")
            m[1].metric("p95", f"{percentile(lat, 95):.2f} s")
            m[2].metric("Within 5 s", f"{sum(x <= 5 for x in lat)}/{len(lat)}")
            st.bar_chart(pd.Series(lat, name="seconds"))
