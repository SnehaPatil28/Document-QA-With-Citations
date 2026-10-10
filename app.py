"""
Streamlit Web Application: Document Q&A with Citations
A local RAG system for querying PDF collections with strict citation verification,
dual chunking strategies (Fixed-size vs. Paragraph-aware), and Ollama LLM integration.
"""

import os
import streamlit as st

# Import core RAG components
from rag_pipeline import (
    DocumentIngestor,
    DocumentIngestionError,
    Chunker,
    VectorStoreManager,
    Retriever,
    OllamaClient,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    TOP_K_DEFAULT,
    MIN_REQUIRED_PAGES
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ===== FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ===== ROOT BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    min-height: 100vh;
}

/* ===== HERO HEADER ===== */
.hero-section {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 40%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto 0.5rem;
    line-height: 1.6;
}

/* ===== DIVIDER ===== */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #6366f1, #818cf8, transparent);
    margin: 1.5rem 0;
    opacity: 0.5;
}

/* ===== METRIC CARDS ===== */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 1.5rem 0;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #818cf8);
    border-radius: 14px 14px 0 0;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.3);
}

.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 6px;
    display: block;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e0e7ff;
    line-height: 1;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ===== QUESTION INPUT AREA ===== */
.query-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(10px);
}

.query-section-title {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 1rem;
}

/* ===== ANSWER BOX ===== */
.answer-wrapper {
    margin: 1.2rem 0;
}

.answer-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.answer-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    flex-shrink: 0;
}

.answer-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #22c55e;
}

.answer-box {
    background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 1rem;
    line-height: 1.75;
    color: #e2e8f0;
    position: relative;
    overflow: hidden;
}

.answer-box::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #22c55e, #16a34a);
    border-radius: 12px 0 0 12px;
}

/* ===== ABSTAIN BOX ===== */
.abstain-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #f97316;
    box-shadow: 0 0 8px #f97316;
    flex-shrink: 0;
}

.abstain-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #f97316;
}

.abstain-box {
    background: rgba(249,115,22,0.06);
    border: 1px solid rgba(249,115,22,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 1rem;
    line-height: 1.75;
    color: #fed7aa;
    position: relative;
}

.abstain-box::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #f97316, #ea580c);
    border-radius: 12px 0 0 12px;
}

.abstain-reason {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(249,115,22,0.15);
}

/* ===== CITATION CARDS ===== */
.citations-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 1rem;
}

.citations-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #818cf8;
}

.cite-count-badge {
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 10px;
}

.citation-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease;
    position: relative;
    overflow: hidden;
}

.citation-card:hover {
    border-color: rgba(99,102,241,0.3);
}

.citation-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #6366f1, #818cf8);
    border-radius: 12px 0 0 12px;
}

.cite-source-id {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    margin-right: 8px;
    letter-spacing: 0.05em;
}

.cite-filename {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e0e7ff;
}

.cite-meta {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
}

.cite-score-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    height: 5px;
    margin: 10px 0 6px;
    overflow: hidden;
}

.cite-score-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #6366f1, #22c55e);
}

.cite-passage {
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #94a3b8;
    white-space: pre-wrap;
    margin-top: 10px;
    line-height: 1.6;
}

/* ===== CANDIDATE PASSAGE ITEM ===== */
.passage-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

.passage-pass  { border-left: 3px solid #22c55e; }
.passage-fail  { border-left: 3px solid #ef4444; }

.passage-score-pass { color: #22c55e; font-weight: 700; font-size: 0.82rem; }
.passage-score-fail { color: #ef4444; font-weight: 700; font-size: 0.82rem; }

.passage-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
}

.pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}

.pill-indigo {
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
}

.pill-slate {
    background: rgba(100,116,139,0.15);
    color: #94a3b8;
    border: 1px solid rgba(100,116,139,0.2);
}

.passage-text-preview {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.77rem;
    color: #64748b;
    white-space: pre-wrap;
    line-height: 1.5;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #cbd5e1 !important;
}

.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6366f1;
    margin: 1.2rem 0 0.5rem;
}

.sidebar-brand {
    text-align: center;
    padding: 1rem 0 0.5rem;
}

.sidebar-brand-name {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e0e7ff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-brand-sub {
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 0.08em;
}

.ollama-connected {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 0.82rem;
    color: #86efac;
}

.ollama-disconnected {
    background: rgba(249,115,22,0.1);
    border: 1px solid rgba(249,115,22,0.25);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 0.82rem;
    color: #fed7aa;
}

/* ===== STREAMLIT OVERRIDES ===== */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    padding: 10px 14px !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #475569 !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

div[data-testid="stExpander"] summary {
    color: #a5b4fc !important;
    font-size: 0.88rem !important;
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #94a3b8 !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.09) !important;
    color: #e2e8f0 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #818cf8) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

/* Warnings / errors */
.stWarning, .stError, .stInfo, .stSuccess {
    border-radius: 10px !important;
}

/* Label text */
label[data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
if "documents_dir" not in st.session_state:
    st.session_state.documents_dir = os.path.join("data", "documents")
if "corpus_pages" not in st.session_state:
    st.session_state.corpus_pages = None
    st.session_state.corpus_stats = None
if "current_strategy" not in st.session_state:
    st.session_state.current_strategy = "Fixed-size (Strategy A)"
if "active_chunks" not in st.session_state:
    st.session_state.active_chunks = None
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-brand'>
        <div class='sidebar-brand-name'>🧠 DocMind</div>
        <div class='sidebar-brand-sub'>RAG · CITATIONS · OLLAMA</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)

    # ── Model Settings ──
    st.markdown("<div class='sidebar-section-title'>⚡ Retrieval Settings</div>", unsafe_allow_html=True)

    selected_strategy_label = st.selectbox(
        "Chunking Strategy",
        options=["Fixed-size (Strategy A)", "Paragraph-aware (Strategy B)"],
        index=0,
        help="Strategy A: ~400 words with ~80 word overlap. Strategy B: paragraph-boundary aware."
    )
    strategy_code = "fixed" if "Fixed-size" in selected_strategy_label else "paragraph"

    sim_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.20, max_value=0.70,
        value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05,
        help="Passages below this cosine similarity are excluded. Lower = more results, higher = more strict."
    )

    top_k = st.slider(
        "Top-K Passages",
        min_value=1, max_value=10,
        value=TOP_K_DEFAULT, step=1,
        help="Number of candidate passages to retrieve from the vector store."
    )

    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)

    # ── Corpus ──
    st.markdown("<div class='sidebar-section-title'>📂 Corpus</div>", unsafe_allow_html=True)
    docs_folder = st.text_input("Documents Directory", value=st.session_state.documents_dir)
    st.session_state.documents_dir = docs_folder

    rebuild_btn = st.button("🔄 Rebuild / Load Index", use_container_width=True)

    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)

    # ── Ollama Status ──
    st.markdown("<div class='sidebar-section-title'>🤖 Ollama Status</div>", unsafe_allow_html=True)
    ollama_client = OllamaClient(model_name=DEFAULT_OLLAMA_MODEL)
    is_ollama_up, ollama_status = ollama_client.check_connection()

    if is_ollama_up:
        st.markdown(f"""
        <div class='ollama-connected'>
            <span>●</span> &nbsp; Connected &nbsp;·&nbsp; <strong>{DEFAULT_OLLAMA_MODEL}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='ollama-disconnected'>
            <span>○</span> &nbsp; Not connected<br>
            <small style='color:#94a3b8;'>Run <code>ollama serve</code> and <code>ollama pull {DEFAULT_OLLAMA_MODEL}</code></small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.7rem;color:#334155;text-align:center;'>Embedding: {DEFAULT_EMBEDDING_MODEL}</div>", unsafe_allow_html=True)


# ── Pipeline Builder ───────────────────────────────────────────────────────────
def get_or_build_pipeline(force_rebuild: bool = False):
    ingestor = DocumentIngestor(documents_dir=st.session_state.documents_dir)
    vector_mgr = VectorStoreManager()

    if st.session_state.corpus_pages is None or force_rebuild:
        with st.spinner("📄 Loading and extracting PDF text…"):
            pages, stats = ingestor.ingest_documents()
            st.session_state.corpus_pages = pages
            st.session_state.corpus_stats = stats

    pages = st.session_state.corpus_pages
    stats = st.session_state.corpus_stats

    cached_data = None if force_rebuild else vector_mgr.load_index(strategy_code)

    if cached_data is not None:
        index, chunks = cached_data
        st.session_state.vector_index = index
        st.session_state.active_chunks = chunks
        st.session_state.current_strategy = selected_strategy_label
    else:
        with st.spinner(f"✂️ Chunking with {selected_strategy_label}…"):
            if strategy_code == "fixed":
                chunks = Chunker.chunk_strategy_a_fixed(pages, target_words=400, overlap_words=80)
            else:
                chunks = Chunker.chunk_strategy_b_paragraph(pages, target_words=400, overlap_words=80)

        with st.spinner(f"🔢 Generating embeddings with {DEFAULT_EMBEDDING_MODEL}…"):
            index, chunks = vector_mgr.build_and_save_index(chunks, strategy=strategy_code)
            st.session_state.vector_index = index
            st.session_state.active_chunks = chunks
            st.session_state.current_strategy = selected_strategy_label

    return ingestor, vector_mgr, stats, st.session_state.active_chunks, st.session_state.vector_index


# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-section'>
    <div class='hero-badge'>Powered by Ollama · RAG · FAISS</div>
    <div class='hero-title'>Document Q&amp;A with Citations</div>
    <div class='hero-subtitle'>
        Ask questions across your PDF corpus. Every answer is grounded in verified passages
        with traceable source citations — no hallucinations.
    </div>
</div>
<div class='glow-divider'></div>
""", unsafe_allow_html=True)


# ── Pipeline Init ──────────────────────────────────────────────────────────────
pipeline_error = None
try:
    strategy_changed = (st.session_state.current_strategy != selected_strategy_label)
    ingestor, vector_mgr, corpus_stats, active_chunks, vector_index = get_or_build_pipeline(
        force_rebuild=(rebuild_btn or strategy_changed)
    )
except DocumentIngestionError as e:
    pipeline_error = str(e)
    st.error(f"⚠️ Document Ingestion Error: {pipeline_error}")
    st.info(
        f"**Troubleshooting:**\n"
        f"- Place PDF files in `{st.session_state.documents_dir}`.\n"
        f"- Total page count must be ≥ {MIN_REQUIRED_PAGES}.\n"
        f"- Scanned PDFs need a text layer (run OCR first)."
    )
    st.stop()
except Exception as e:
    pipeline_error = str(e)
    st.error(f"⚠️ Pipeline Error: {pipeline_error}")
    st.stop()


# ── Metrics Row ────────────────────────────────────────────────────────────────
strategy_emoji = "🔷" if strategy_code == "fixed" else "🔶"
c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("📄", corpus_stats["total_pages"], "Total Pages"),
    ("✅", corpus_stats["extracted_pages"], "Extracted Pages"),
    (strategy_emoji, len(active_chunks), f"{selected_strategy_label.split()[0]} Chunks"),
    ("📁", corpus_stats["total_files"], "PDF Documents"),
]
for col, (icon, value, label) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <span class='metric-icon'>{icon}</span>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

if corpus_stats.get("ocr_warnings"):
    for warn in corpus_stats["ocr_warnings"]:
        st.warning(f"⚠️ {warn}")

st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)


# ── Query Section ──────────────────────────────────────────────────────────────
st.markdown("<div class='query-section-title'>💬 Ask a Question</div>", unsafe_allow_html=True)

sample_questions = [
    "What are the four core functions of the NIST AI Risk Management Framework?",
    "How does the framework define trustworthy AI systems?",
    "What specific risks are introduced by generative artificial intelligence according to NIST AI 600-1?",
    "What ingredients are needed to bake an authentic Neapolitan pizza?",
]

col_sample, col_spacer = st.columns([3, 1])
with col_sample:
    selected_sample = st.selectbox(
        "Quick-select a sample question:",
        options=["— Type your own below —"] + sample_questions,
        index=0,
        label_visibility="collapsed"
    )

default_query = selected_sample if selected_sample != "— Type your own below —" else ""

user_question = st.text_input(
    "Your question",
    value=default_query,
    placeholder="e.g. What are the key risk categories for generative AI?",
    label_visibility="collapsed",
    key="user_query_input"
)

col_ask, col_clear, col_pad = st.columns([2, 1, 5])
with col_ask:
    ask_clicked = st.button("⚡ Ask DocMind", type="primary", use_container_width=True)
with col_clear:
    clear_clicked = st.button("✕ Clear", type="secondary", use_container_width=True)

if clear_clicked:
    st.session_state.last_result = None
    st.session_state.last_question = ""
    st.rerun()

st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)


# ── Answer Processing ──────────────────────────────────────────────────────────
if ask_clicked and user_question.strip():
    st.session_state.last_question = user_question.strip()
    st.session_state.last_result = None  # reset while fetching

    retriever = Retriever(
        vector_store_mgr=vector_mgr,
        index=vector_index,
        chunks=active_chunks,
        similarity_threshold=sim_threshold
    )

    with st.spinner("🔍 Retrieving relevant passages…"):
        retrieved_passages = retriever.retrieve(user_question, top_k=top_k)

    if not retrieved_passages:
        st.session_state.last_result = {
            "type": "abstain",
            "answer": OllamaClient.ABSTAIN_RESPONSE,
            "reason": "No passages retrieved from the vector store.",
            "passages": []
        }
    else:
        top_score = retrieved_passages[0]["score"]
        qualifying = [p for p in retrieved_passages if p["score"] >= sim_threshold]

        if not qualifying:
            st.session_state.last_result = {
                "type": "abstain",
                "answer": OllamaClient.ABSTAIN_RESPONSE,
                "reason": (
                    f"Top passage scored {top_score:.2f}, below threshold {sim_threshold:.2f}. "
                    "Abstained to prevent hallucinations."
                ),
                "passages": retrieved_passages
            }
        else:
            with st.spinner(f"🤖 Generating answer with {DEFAULT_OLLAMA_MODEL}…"):
                try:
                    result = ollama_client.generate_answer(
                        question=user_question,
                        retrieved_passages=retrieved_passages,
                        similarity_threshold=sim_threshold
                    )
                    result["passages"] = retrieved_passages
                    result["type"] = "abstain" if result["is_abstained"] else "answer"
                    result["reason"] = result.get("abstain_reason", "")
                    st.session_state.last_result = result
                except ConnectionError as ce:
                    st.error(f"❌ Ollama Connection Error: {str(ce)}")
                    st.info(
                        "**Fix:**\n"
                        "1. Open a terminal → `ollama serve`\n"
                        f"2. Pull model: `ollama pull {DEFAULT_OLLAMA_MODEL}`\n"
                        "3. Try again."
                    )
                except Exception as ex:
                    st.error(f"❌ Generation error: {str(ex)}")


# ── Render Stored Result ───────────────────────────────────────────────────────
result = st.session_state.get("last_result")
question = st.session_state.get("last_question", "")

if result:
    # ── Question Echo ──
    st.markdown(f"""
    <div style='margin-bottom:1rem;'>
        <span style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:#475569;font-weight:600;'>Question</span><br>
        <span style='font-size:1.05rem;color:#e0e7ff;font-weight:500;'>"{question}"</span>
    </div>
    """, unsafe_allow_html=True)

    if result["type"] == "abstain":
        # ── Abstain Box ──
        st.markdown(f"""
        <div class='answer-wrapper'>
            <div class='answer-header'>
                <div class='abstain-dot'></div>
                <span class='abstain-label'>No Sufficient Evidence</span>
            </div>
            <div class='abstain-box'>
                {result['answer']}
                <div class='abstain-reason'>ℹ️ {result.get('reason', '')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Answer Box ──
        st.markdown(f"""
        <div class='answer-wrapper'>
            <div class='answer-header'>
                <div class='answer-dot'></div>
                <span class='answer-label'>Answer</span>
            </div>
            <div class='answer-box'>
                {result['answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Citations ──
        citations = result.get("citations", [])
        if citations:
            st.markdown(f"""
            <div class='citations-header'>
                <span class='citations-title'>📌 Verified Source Citations</span>
                <span class='cite-count-badge'>{len(citations)} sources</span>
            </div>
            """, unsafe_allow_html=True)

            for cite in citations:
                score_pct = min(int(cite["score"] * 100), 100)
                score_color = "#22c55e" if cite["score"] >= 0.5 else "#f59e0b" if cite["score"] >= 0.35 else "#ef4444"
                passage_preview = cite["text"][:600] + ("…" if len(cite["text"]) > 600 else "")
                st.markdown(f"""
                <div class='citation-card'>
                    <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;'>
                        <span class='cite-source-id'>{cite['source_id']}</span>
                        <span class='cite-filename'>{cite['filename']}</span>
                        <span class='pill pill-slate'>Page {cite['page_number']}</span>
                        <span style='margin-left:auto;font-size:0.78rem;font-weight:700;color:{score_color};'>
                            {cite['score']:.4f} similarity
                        </span>
                    </div>
                    <div class='cite-score-bar-bg'>
                        <div class='cite-score-bar' style='width:{score_pct}%;background:linear-gradient(90deg,#6366f1,{score_color});'></div>
                    </div>
                    <div class='cite-passage'>{passage_preview}</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("The model did not include standard [S#] citations in its response.")

        if result.get("invalid_citations"):
            st.warning(f"⚠️ Model referenced non-existent citation IDs: {result['invalid_citations']}")

    # ── Candidate Passages ──
    passages = result.get("passages", [])
    if passages:
        with st.expander(f"🔎 View All {len(passages)} Retrieved Candidate Passages (ranked by similarity)"):
            for p in passages:
                status_class = "passage-pass" if p["meets_threshold"] else "passage-fail"
                score_class = "passage-score-pass" if p["meets_threshold"] else "passage-score-fail"
                threshold_pill_color = "pill-indigo" if p["meets_threshold"] else "pill-slate"
                threshold_label = "✓ above threshold" if p["meets_threshold"] else "✗ below threshold"
                text_preview = p["text"][:350] + ("…" if len(p["text"]) > 350 else "")
                st.markdown(f"""
                <div class='passage-item {status_class}'>
                    <div class='passage-meta-row'>
                        <span class='pill pill-indigo'>{p['source_id']}</span>
                        <span class='{score_class}'>Score: {p['score']}</span>
                        <span class='pill pill-slate'>{p['filename']}</span>
                        <span class='pill pill-slate'>Page {p['page_number']}</span>
                        <span class='pill {threshold_pill_color}'>{threshold_label}</span>
                        <span class='pill pill-slate'>{p['word_count']} words</span>
                    </div>
                    <div class='passage-text-preview'>{text_preview}</div>
                </div>
                """, unsafe_allow_html=True)

elif not question:
    # ── Empty State ──
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem;'>
        <div style='font-size:3.5rem;margin-bottom:1rem;'>🧠</div>
        <div style='font-size:1.1rem;font-weight:600;color:#475569;margin-bottom:0.4rem;'>Ready to answer your questions</div>
        <div style='font-size:0.9rem;color:#334155;max-width:400px;margin:0 auto;'>
            Select a sample question or type your own above, then click <strong style='color:#818cf8;'>⚡ Ask DocMind</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
