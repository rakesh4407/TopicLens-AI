"""
app.py
TopicLens AI — Intelligent Text Topic Modelling & Discovery Platform
BCA AI & Data Science Capstone Project #9
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.session_manager import (
    init_session_state,
    reset_pipeline,
    has_dataset,
    has_cleaned_data,
    has_trained_model,
)
from src.views.home_view import render_home_view
from src.views.dataset_view import render_dataset_view
from src.views.preprocessing_view import render_preprocessing_view
from src.views.topic_modelling_view import render_topic_modelling_view
from src.views.topic_exploration_view import render_topic_exploration_view
from src.views.document_analysis_view import render_document_analysis_view
from src.views.visual_analytics_view import render_visual_analytics_view
from src.views.model_evaluation_view import render_model_evaluation_view
from src.views.export_results_view import render_export_results_view
from src.views.about_project_view import render_about_project_view

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="TopicLens AI — Text Topic Modelling",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS Design System ─────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>

/* ══════════════════════════════════════════════════════
   GLOBAL RESET & TYPOGRAPHY
══════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ══════════════════════════════════════════════════════
   MAIN CONTENT AREA
══════════════════════════════════════════════════════ */
.main .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* ══════════════════════════════════════════════════════
   SIDEBAR — Dark Glass Theme
══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1a2744 50%, #0f172a 100%) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.15) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stSidebarContent"] {
    background: transparent !important;
}

/* ══════════════════════════════════════════════════════
   BRAND HEADER
══════════════════════════════════════════════════════ */
.brand-header {
    padding: 24px 20px 18px 20px;
    background: linear-gradient(135deg, rgba(37,99,235,0.3) 0%, rgba(124,58,237,0.2) 100%);
    border-bottom: 1px solid rgba(99,179,237,0.2);
    margin-bottom: 8px;
}
.brand-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.brand-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4);
}
.brand-name {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px;
    line-height: 1.1;
}
.brand-tagline {
    font-size: 0.72rem;
    color: rgba(148, 163, 184, 0.9);
    letter-spacing: 0.3px;
    margin-top: 2px;
}
.brand-badge {
    display: inline-block;
    padding: 2px 8px;
    background: linear-gradient(135deg, rgba(37,99,235,0.3), rgba(124,58,237,0.3));
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 600;
    color: #93C5FD;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ══════════════════════════════════════════════════════
   NAVIGATION SECTION LABEL
══════════════════════════════════════════════════════ */
.nav-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: rgba(100,116,139,0.9);
    padding: 12px 20px 4px 20px;
}

/* ══════════════════════════════════════════════════════
   RADIO NAV BUTTONS — Override Streamlit default
══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 20px !important;
    margin: 0 !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: rgba(203, 213, 225, 0.85) !important;
    border: 1px solid transparent !important;
    gap: 10px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
    background: rgba(99,179,237,0.08) !important;
    color: #E2E8F0 !important;
    border-color: rgba(99,179,237,0.15) !important;
    padding-left: 24px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label[data-baseweb="radio"] input:checked ~ div,
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label[aria-checked="true"] {
    background: linear-gradient(135deg, rgba(37,99,235,0.25), rgba(124,58,237,0.2)) !important;
    color: #60A5FA !important;
    border-color: rgba(37,99,235,0.4) !important;
    font-weight: 600 !important;
}
/* Hide the radio circle */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.875rem !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label > div:first-child {
    display: none !important;
}

/* ══════════════════════════════════════════════════════
   PIPELINE STATUS CARDS
══════════════════════════════════════════════════════ */
.pipeline-container {
    padding: 14px 16px;
    margin: 8px 12px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 12px;
    backdrop-filter: blur(10px);
}
.pipeline-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: rgba(100,116,139,0.9);
    margin-bottom: 10px;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(99,179,237,0.07);
}
.pipeline-step:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.step-dot-done {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #10B981, #34D399);
    box-shadow: 0 0 6px rgba(16,185,129,0.5);
    flex-shrink: 0;
}
.step-dot-pending {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(100,116,139,0.4);
    border: 1px solid rgba(100,116,139,0.3);
    flex-shrink: 0;
}
.step-label {
    font-size: 0.78rem;
    font-weight: 500;
    flex: 1;
}
.step-label-done { color: #94A3B8; }
.step-label-pending { color: rgba(100,116,139,0.7); }
.step-badge-done {
    font-size: 0.6rem;
    font-weight: 700;
    padding: 2px 7px;
    background: rgba(16,185,129,0.15);
    color: #34D399;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    letter-spacing: 0.3px;
}
.step-badge-pending {
    font-size: 0.6rem;
    font-weight: 600;
    padding: 2px 7px;
    background: rgba(100,116,139,0.1);
    color: rgba(100,116,139,0.6);
    border: 1px solid rgba(100,116,139,0.15);
    border-radius: 20px;
}
.step-badge-active {
    font-size: 0.6rem;
    font-weight: 700;
    padding: 2px 7px;
    background: rgba(37,99,235,0.2);
    color: #60A5FA;
    border: 1px solid rgba(37,99,235,0.35);
    border-radius: 20px;
    animation: pulse-badge 2s infinite;
}
@keyframes pulse-badge {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ══════════════════════════════════════════════════════
   SIDEBAR RESET BUTTON
══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.1)) !important;
    color: #FCA5A5 !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(239,68,68,0.3), rgba(220,38,38,0.2)) !important;
    border-color: rgba(239,68,68,0.5) !important;
    color: #FECACA !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(239,68,68,0.2) !important;
}

/* ══════════════════════════════════════════════════════
   SIDEBAR FOOTER
══════════════════════════════════════════════════════ */
.sidebar-footer {
    padding: 12px 20px;
    border-top: 1px solid rgba(99,179,237,0.1);
    margin-top: 8px;
}
.sidebar-footer-text {
    font-size: 0.65rem;
    color: rgba(100,116,139,0.6);
    text-align: center;
    line-height: 1.6;
}

/* ══════════════════════════════════════════════════════
   MAIN PAGE METRIC CARDS
══════════════════════════════════════════════════════ */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #F8FAFF 0%, #EFF6FF 100%);
    border: 1px solid #BFDBFE;
    padding: 16px 20px !important;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(37,99,235,0.08), 0 4px 16px rgba(37,99,235,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(37,99,235,0.12);
}
div[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: #3B82F6 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #1E3A8A !important;
    line-height: 1.1 !important;
}

/* ══════════════════════════════════════════════════════
   GENERAL BUTTONS
══════════════════════════════════════════════════════ */
.stButton > button[kind="primary"],
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
}

/* ══════════════════════════════════════════════════════
   DATAFRAME & TABLE
══════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ══════════════════════════════════════════════════════
   EXPANDER
══════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #F8FAFC !important;
}
[data-testid="stExpander"]:hover {
    border-color: #BFDBFE !important;
}

/* ══════════════════════════════════════════════════════
   TABS
══════════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-testid="stTab"] {
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 8px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    color: #64748B !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
}

/* ══════════════════════════════════════════════════════
   ALERTS / INFO BOXES
══════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-size: 0.9rem !important;
}

/* ══════════════════════════════════════════════════════
   SCROLLBAR STYLING
══════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.5); }

/* ══════════════════════════════════════════════════════
   HIDE STREAMLIT DEFAULT DECORATIONS
══════════════════════════════════════════════════════ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ─────────────────────────────────────────────────
init_session_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Brand Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">
            <div class="brand-icon">🔍</div>
            <div>
                <div class="brand-name">TopicLens AI</div>
                <div class="brand-tagline">Intelligent Topic Discovery Platform</div>
            </div>
        </div>
        <div class="brand-badge">BCA · AI & Data Science · Project #9</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

    PAGES = [
        "🏠 Home Dashboard",
        "📂 Dataset Upload",
        "🧹 Data Preprocessing",
        "⚙️ Topic Modelling",
        "🔍 Topic Exploration",
        "📄 Document Analysis",
        "📊 Visual Analytics",
        "📈 Model Evaluation",
        "💾 Export Results",
        "ℹ️ About Project",
    ]

    default_index = 0
    if st.session_state.get("current_page") in PAGES:
        default_index = PAGES.index(st.session_state["current_page"])

    selected_page = st.radio(
        "Navigation",
        options=PAGES,
        index=default_index,
        label_visibility="collapsed",
    )
    st.session_state["current_page"] = selected_page

    # ── Pipeline Status ───────────────────────────────────────────────────────
    ds_ok   = has_dataset()
    prep_ok = has_cleaned_data()
    tm      = st.session_state.get("topic_modeler")
    model_ok = bool(tm and tm.is_fitted)

    def _step(label, done, badge_done="DONE", badge_pending="PENDING", active=False):
        dot   = 'step-dot-done'   if done else 'step-dot-pending'
        lbl   = 'step-label-done' if done else 'step-label-pending'
        badge = f'step-badge-done" >{badge_done}' if done else (
                f'step-badge-active" >{badge_pending}' if active else
                f'step-badge-pending" >{badge_pending}')
        return f"""
        <div class="pipeline-step">
            <div class="{dot}"></div>
            <span class="{lbl}">{label}</span>
            <span class="{badge}</span>
        </div>"""

    k_str = f"K={tm.n_topics}" if model_ok else "TRAIN"

    pipeline_html = f"""
    <div class="pipeline-container">
        <div class="pipeline-title">⚡ Pipeline Status</div>
        {_step("Dataset Loaded", ds_ok,   badge_pending="UPLOAD", active=not ds_ok)}
        {_step("Preprocessing", prep_ok,  badge_pending="CLEAN",  active=ds_ok and not prep_ok)}
        {_step("LDA Model",     model_ok, badge_done=k_str, badge_pending="TRAIN", active=prep_ok and not model_ok)}
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # ── Reset ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset Pipeline", use_container_width=True):
        reset_pipeline()
        st.session_state["current_page"] = "🏠 Home Dashboard"
        st.success("Pipeline reset successfully.")
        st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-text">
            TopicLens AI · v2.0<br>
            Text Topic Modelling · LDA<br>
            BCA Capstone · 2024
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Page Router ──────────────────────────────────────────────────────────────
if   selected_page == "🏠 Home Dashboard":    render_home_view()
elif selected_page == "📂 Dataset Upload":    render_dataset_view()
elif selected_page == "🧹 Data Preprocessing": render_preprocessing_view()
elif selected_page == "⚙️ Topic Modelling":   render_topic_modelling_view()
elif selected_page == "🔍 Topic Exploration":  render_topic_exploration_view()
elif selected_page == "📄 Document Analysis":  render_document_analysis_view()
elif selected_page == "📊 Visual Analytics":   render_visual_analytics_view()
elif selected_page == "📈 Model Evaluation":   render_model_evaluation_view()
elif selected_page == "💾 Export Results":     render_export_results_view()
elif selected_page == "ℹ️ About Project":      render_about_project_view()
