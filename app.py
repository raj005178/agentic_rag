"""
app.py — Streamlit frontend for the Agentic RAG system.

Run:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from pipeline.loaders import load_file, SUPPORTED_TYPES
from pipeline.chunker import chunk_document
from pipeline.retriever import ingest_document, get_doc_count, get_chunks_per_doc, clear_collection
from pipeline.graph import run_query
from pipeline.llm import stream_generate
from evaluator import evaluate

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Dark slate background with indigo accent */
:root {
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --border:   #2e3148;
    --accent:   #6366f1;
    --accent2:  #818cf8;
    --text:     #e2e8f0;
    --muted:    #94a3b8;
    --green:    #4ade80;
    --amber:    #fbbf24;
    --red:      #f87171;
}

.source-card {
    background: var(--surface);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: var(--text);
}
.source-card b { color: var(--accent2); }
.source-card small { color: var(--muted); }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-green  { background: #14532d; color: var(--green); }
.badge-amber  { background: #78350f; color: var(--amber); }
.badge-red    { background: #7f1d1d; color: var(--red);   }
.badge-indigo { background: #312e81; color: var(--accent2); }

.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 8px 0;
}
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    min-width: 110px;
    text-align: center;
}
.metric-box .val {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent2);
}
.metric-box .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "messages":             [],
        "conversation_history": [],
        "uploaded_files":       [],
        "last_metrics":         None,
        "last_result":          None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Agentic RAG")
    st.caption("Private · Verifiable · Self-correcting")
    st.divider()

    # Upload
    st.markdown("### 📄 Documents")
    ext_list = "  ".join(f"`{e}`" for e in SUPPORTED_TYPES)
    st.caption(f"Supported: {ext_list}")

    uploaded = st.file_uploader(
        "Upload files",
        type=[e.lstrip(".") for e in SUPPORTED_TYPES],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    chunking_strategy = st.selectbox(
        "Chunking strategy",
        ["parent_child", "semantic", "sentence", "fixed"],
        index=0,
        help=(
            "parent_child — best precision/context balance (recommended)\n"
            "semantic     — split on topic shifts\n"
            "sentence     — natural sentence boundaries\n"
            "fixed        — fixed character windows"
        ),
    )

    if uploaded and st.button("⬆️ Ingest", type="primary", use_container_width=True):
        new_files = [f for f in uploaded if f.name not in st.session_state.uploaded_files]
        if not new_files:
            st.info("All files already ingested.")
        else:
            progress = st.progress(0)
            for i, f in enumerate(new_files):
                with st.spinner(f"Processing {f.name} …"):
                    ext = "." + f.name.rsplit(".", 1)[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        doc = load_file(tmp_path)
                        chunks = chunk_document(doc["text"], strategy=chunking_strategy)
                        count = ingest_document(chunks, f.name)
                        st.session_state.uploaded_files.append(f.name)
                        st.success(f"✅ {f.name} → {count} chunks")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
                    finally:
                        os.unlink(tmp_path)
                progress.progress((i + 1) / len(new_files))

    st.divider()

    # Stats
    chunk_count = get_doc_count()
    st.markdown("### 📊 Store")
    c1, c2 = st.columns(2)
    c1.metric("Total Chunks", chunk_count)
    c2.metric("Files", len(st.session_state.uploaded_files))

    # Per-document chunk counts
    chunks_per_doc = get_chunks_per_doc()
    if chunks_per_doc:
        for fname, count in chunks_per_doc.items():
            st.caption(f"• {fname} → {count} chunks")

    st.divider()

    col_a, col_b = st.columns(2)
    if col_a.button("🗑 Clear docs", use_container_width=True):
        clear_collection()
        st.session_state.uploaded_files = []
        st.rerun()
    if col_b.button("💬 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()


# ── Main area — two tabs ──────────────────────────────────────────────────────
tab_chat, tab_metrics = st.tabs(["💬  Chat", "📈  Metrics"])


# ── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # Inline badges
                meta = msg.get("meta", {})
                conf = meta.get("confidence", 0.0)
                intent = meta.get("intent", "—")
                model = meta.get("model_used", "—")
                web = meta.get("web_search_used", False)

                conf_cls = (
                    "badge-green" if conf > 0.70
                    else "badge-amber" if conf > 0.40
                    else "badge-red"
                )
                web_badge = '<span class="badge badge-indigo">🌐 Web</span>' if web else ""
                st.markdown(
                    f'<span class="badge {conf_cls}">Confidence {conf:.0%}</span>'
                    f'<span class="badge badge-indigo">{intent}</span>'
                    f'<span class="badge badge-indigo">{model}</span>'
                    f'{web_badge}',
                    unsafe_allow_html=True,
                )

                # Sources
                if msg.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        for src in msg["sources"]:
                            rel = src["relevance"]
                            rel_cls = (
                                "badge-green" if rel > 0.7
                                else "badge-amber" if rel > 0.4
                                else "badge-red"
                            )
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>{src["source"]}</b>'
                                f' <span class="badge {rel_cls}">Score {rel:.2f}</span>'
                                f'<br><small>{src["snippet"]} …</small>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

    # Input
    if prompt := st.chat_input("Ask anything about your documents …"):
        if chunk_count == 0:
            st.warning("⚠️ Upload and ingest at least one document first.")
            st.stop()

        # User bubble
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant bubble — stream the final answer
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            status_placeholder = st.empty()

            status_placeholder.caption("⚙️ Running agentic pipeline …")
            result = run_query(prompt, st.session_state.conversation_history)
            answer = result["final_answer"]
            sources = result.get("sources", [])
            metrics = result.get("metrics", {})

            # Stream the answer token by token (re-generates for streaming effect)
            streamed = ""
            for chunk in stream_generate(
                f"Restate this answer fluently:\n{answer}", max_tokens=900
            ):
                streamed += chunk
                answer_placeholder.markdown(streamed + "▌")
            answer_placeholder.markdown(streamed)
            status_placeholder.empty()

            # Use the pipeline answer as ground truth (not re-generated)
            final_displayed = answer

            # Badges
            conf = result.get("confidence_score", 0.0)
            conf_cls = (
                "badge-green" if conf > 0.70
                else "badge-amber" if conf > 0.40
                else "badge-red"
            )
            web = metrics.get("web_search_used", False)
            web_badge = '<span class="badge badge-indigo">🌐 Web used</span>' if web else ""
            st.markdown(
                f'<span class="badge {conf_cls}">Confidence {conf:.0%}</span>'
                f'<span class="badge badge-indigo">{metrics.get("intent","—")}</span>'
                f'<span class="badge badge-indigo">{metrics.get("model_used","—")}</span>'
                f"{web_badge}",
                unsafe_allow_html=True,
            )

            # Sources
            if sources:
                with st.expander("📚 Sources", expanded=True):
                    for src in sources:
                        rel = src["relevance"]
                        rel_cls = (
                            "badge-green" if rel > 0.7
                            else "badge-amber" if rel > 0.4
                            else "badge-red"
                        )
                        st.markdown(
                            f'<div class="source-card">'
                            f'<b>{src["source"]}</b>'
                            f' <span class="badge {rel_cls}">Score {rel:.2f}</span>'
                            f'<br><small>{src["snippet"]} …</small>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        # Update histories
        st.session_state.conversation_history.extend([
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": final_displayed},
        ])
        # Keep last 10 turns
        st.session_state.conversation_history = (
            st.session_state.conversation_history[-10:]
        )

        # Compute RAGAS metrics in background
        ragas = evaluate(prompt, final_displayed, result.get("graded_docs", []))
        metrics.update(ragas)
        st.session_state.last_metrics = metrics
        st.session_state.last_result = result

        st.session_state.messages.append({
            "role":    "assistant",
            "content": final_displayed,
            "sources": sources,
            "meta":    metrics,
        })
        # Force metrics tab to refresh
        st.rerun()


# ── Metrics tab ───────────────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("### 📈 Last Query — Evaluation Dashboard")

    if not st.session_state.last_metrics:
        st.info("Ask a question in the Chat tab to see metrics here.")
        st.stop()

    m = st.session_state.last_metrics

    def _pct(key: str) -> str:
        return f"{m.get(key, 0):.0%}"

    def _val(key: str, default="—"):
        return m.get(key, default)

    # RAG quality
    st.markdown("#### RAG Quality")
    st.markdown(
        f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{_pct("groundedness")}</div><div class="lbl">Groundedness</div></div>
  <div class="metric-box"><div class="val">{_pct("confidence")}</div><div class="lbl">Confidence</div></div>
  <div class="metric-box"><div class="val">{_pct("answer_relevance")}</div><div class="lbl">Answer Relevance</div></div>
  <div class="metric-box"><div class="val">{_pct("faithfulness")}</div><div class="lbl">Faithfulness</div></div>
  <div class="metric-box"><div class="val">{_pct("context_recall")}</div><div class="lbl">Context Recall</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    # Pipeline telemetry
    st.markdown("#### Pipeline Telemetry")
    st.markdown(
        f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{_val("docs_retrieved", 0)}</div><div class="lbl">Docs Retrieved</div></div>
  <div class="metric-box"><div class="val">{_val("docs_after_grade", 0)}</div><div class="lbl">After Grading</div></div>
  <div class="metric-box"><div class="val">{_val("retry_count", 0)}</div><div class="lbl">Retries</div></div>
  <div class="metric-box"><div class="val">{"✅" if _val("web_search_used") else "—"}</div><div class="lbl">Web Search</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    # Model & intent
    st.markdown("#### Decision Details")
    c1, c2 = st.columns(2)
    c1.info(f"**Intent detected:** `{_val('intent')}`")
    c2.info(f"**Model used:** `{_val('model_used')}`")

    # Tree of Thought drafts
    result = st.session_state.last_result
    if result and result.get("draft_answers"):
        drafts = result["draft_answers"]
        draft_label = f"🌳 Tree of Thought — {len(drafts)} Draft{'s' if len(drafts) > 1 else ''}"
        intent = result.get("metrics", {}).get("intent", "")
        if intent == "FACTUAL":
            draft_label += " (FACTUAL — single draft mode)"
        with st.expander(draft_label):
            for i, draft in enumerate(drafts, 1):
                st.markdown(f"**Draft {i}**")
                st.text_area(
                    f"draft_{i}",
                    value=draft,
                    height=120,
                    label_visibility="collapsed"
                )
