"""
app.py
TopicLens AI — Intelligent Text Topic Modelling & Discovery Platform
BCA AI & Data Science Capstone Project #9
"""

import sys
from pathlib import Path

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

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="TopicLens AI — Text Topic Modelling",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injected as a single clean <style> block (no <link> tags) ─────────────
_CSS = """
<style>
/* ── GLOBAL ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.main .block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}
/* ── SIDEBAR ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2744 55%, #0f172a 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15) !important;
}
[data-testid="stSidebarContent"] {
    background: transparent !important;
}
/* ── BRAND ────────────────────────────────────────────── */
.tl-brand {
    padding: 20px 18px 16px 18px;
    background: linear-gradient(135deg,rgba(37,99,235,.25),rgba(124,58,237,.18));
    border-bottom: 1px solid rgba(99,179,237,.2);
    margin-bottom: 6px;
}
.tl-brand-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
}
.tl-brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 14px rgba(37,99,235,.4);
    flex-shrink: 0;
}
.tl-brand-name {
    font-size: 1.2rem;
    font-weight: 800;
    background: linear-gradient(135deg,#60a5fa,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -.3px;
    line-height: 1.1;
}
.tl-brand-tag {
    font-size: .7rem;
    color: rgba(148,163,184,.85);
    letter-spacing: .3px;
}
.tl-brand-pill {
    display: inline-block;
    padding: 2px 9px;
    background: linear-gradient(135deg,rgba(37,99,235,.25),rgba(124,58,237,.25));
    border: 1px solid rgba(99,179,237,.3);
    border-radius: 20px;
    font-size: .62rem;
    font-weight: 700;
    color: #93c5fd;
    letter-spacing: .5px;
    text-transform: uppercase;
}
/* ── NAV LABEL ────────────────────────────────────────── */
.tl-nav-label {
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: rgba(100,116,139,.8);
    padding: 10px 18px 3px 18px;
}
/* ── RADIO (hide circles, style labels) ───────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display:none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:1px !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
    display: flex !important;
    align-items: center !important;
    padding: 8px 18px !important;
    margin: 0 !important;
    border-radius: 7px !important;
    cursor: pointer !important;
    font-size: .855rem !important;
    font-weight: 500 !important;
    color: rgba(203,213,225,.8) !important;
    border: 1px solid transparent !important;
    transition: all .15s ease !important;
    gap: 8px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
    background: rgba(99,179,237,.09) !important;
    color: #e2e8f0 !important;
    border-color: rgba(99,179,237,.2) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label > div:first-child {
    display: none !important;
}
/* ── PIPELINE STATUS ──────────────────────────────────── */
.tl-pipe {
    padding: 12px 14px;
    margin: 6px 10px;
    background: rgba(10,20,38,.55);
    border: 1px solid rgba(99,179,237,.12);
    border-radius: 10px;
}
.tl-pipe-title {
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: rgba(100,116,139,.85);
    margin-bottom: 8px;
}
.tl-step {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 6px 0;
    border-bottom: 1px solid rgba(99,179,237,.06);
}
.tl-step:last-child { border-bottom: none; }
.tl-dot-g {
    width: 7px; height: 7px; border-radius: 50%;
    background: linear-gradient(135deg,#10b981,#34d399);
    box-shadow: 0 0 6px rgba(16,185,129,.5);
    flex-shrink: 0;
}
.tl-dot-b {
    width: 7px; height: 7px; border-radius: 50%;
    background: linear-gradient(135deg,#2563eb,#60a5fa);
    box-shadow: 0 0 6px rgba(37,99,235,.5);
    flex-shrink: 0;
    animation: tl-pulse 1.8s infinite;
}
.tl-dot-x {
    width: 7px; height: 7px; border-radius: 50%;
    background: rgba(100,116,139,.35);
    border: 1px solid rgba(100,116,139,.25);
    flex-shrink: 0;
}
@keyframes tl-pulse { 0%,100%{opacity:1;} 50%{opacity:.4;} }
.tl-step-lbl { font-size: .77rem; font-weight: 500; flex: 1; }
.tl-lbl-g { color: #94a3b8; }
.tl-lbl-b { color: #60a5fa; font-weight: 600; }
.tl-lbl-x { color: rgba(100,116,139,.65); }
.tl-badge-g {
    font-size: .58rem; font-weight: 700; padding: 2px 7px;
    background: rgba(16,185,129,.15); color: #34d399;
    border: 1px solid rgba(16,185,129,.3); border-radius: 20px;
}
.tl-badge-b {
    font-size: .58rem; font-weight: 700; padding: 2px 7px;
    background: rgba(37,99,235,.2); color: #60a5fa;
    border: 1px solid rgba(37,99,235,.35); border-radius: 20px;
    animation: tl-pulse 1.8s infinite;
}
.tl-badge-x {
    font-size: .58rem; font-weight: 600; padding: 2px 7px;
    background: rgba(100,116,139,.1); color: rgba(100,116,139,.55);
    border: 1px solid rgba(100,116,139,.15); border-radius: 20px;
}
/* ── RESET BUTTON ─────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg,rgba(239,68,68,.15),rgba(220,38,38,.1)) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,.3) !important;
    border-radius: 8px !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
    transition: all .2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg,rgba(239,68,68,.3),rgba(220,38,38,.2)) !important;
    color: #fecaca !important;
    border-color: rgba(239,68,68,.5) !important;
    box-shadow: 0 4px 12px rgba(239,68,68,.2) !important;
}
/* ── SIDEBAR FOOTER ────────────────────────────────────── */
.tl-footer {
    padding: 10px 18px;
    border-top: 1px solid rgba(99,179,237,.1);
    margin-top: 6px;
    text-align: center;
    font-size: .62rem;
    color: rgba(100,116,139,.55);
    line-height: 1.7;
}
/* ── MAIN METRIC CARDS ────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg,#f8faff,#eff6ff) !important;
    border: 1px solid #bfdbfe !important;
    padding: 16px 20px !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(37,99,235,.07) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,.12) !important;
}
div[data-testid="stMetric"] label {
    font-size: .72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .5px !important;
    color: #3b82f6 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: #1e3a8a !important;
    line-height: 1.1 !important;
}
/* ── BUTTONS ──────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#2563eb,#7c3aed) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.35) !important;
    transition: all .2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,.45) !important;
}
/* ── MISC ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius:10px !important; overflow:hidden !important; }
[data-testid="stExpander"] { border-radius:10px !important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-thumb { background:rgba(99,179,237,.35); border-radius:3px; }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }
header { visibility:hidden; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
init_session_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown("""
<div class="tl-brand">
  <div class="tl-brand-row">
    <div class="tl-brand-icon">🔍</div>
    <div>
      <div class="tl-brand-name">TopicLens AI</div>
      <div class="tl-brand-tag">Intelligent Topic Discovery Platform</div>
    </div>
  </div>
  <div class="tl-brand-pill">BCA &middot; AI &amp; Data Science &middot; Project #9</div>
</div>
""", unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="tl-nav-label">Navigation</div>', unsafe_allow_html=True)

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

    # Pipeline status
    ds_ok    = has_dataset()
    prep_ok  = has_cleaned_data()
    tm       = st.session_state.get("topic_modeler")
    model_ok = bool(tm and getattr(tm, "is_fitted", False))

    def _step_html(label, state):
        # state: "done" | "active" | "wait"
        dot   = {"done": "tl-dot-g", "active": "tl-dot-b", "wait": "tl-dot-x"}[state]
        lbl   = {"done": "tl-lbl-g", "active": "tl-lbl-b", "wait": "tl-lbl-x"}[state]
        badge = {"done": "tl-badge-g", "active": "tl-badge-b", "wait": "tl-badge-x"}[state]
        text  = {"done": "DONE", "active": "ACTIVE", "wait": "PENDING"}[state]
        return (
            f'<div class="tl-step">'
            f'<div class="{dot}"></div>'
            f'<span class="tl-step-lbl {lbl}">{label}</span>'
            f'<span class="{badge}">{text}</span>'
            f'</div>'
        )

    k_label = f"K={tm.n_topics}" if model_ok else "TRAIN"

    ds_state    = "done"   if ds_ok    else "active"
    prep_state  = "done"   if prep_ok  else ("active" if ds_ok    else "wait")
    model_state = "done"   if model_ok else ("active" if prep_ok  else "wait")

    pipe_html = (
        '<div class="tl-pipe">'
        '<div class="tl-pipe-title">⚡ Pipeline Status</div>'
        + _step_html("Dataset Loaded", ds_state)
        + _step_html("Preprocessing",  prep_state)
        + _step_html(f"LDA Model ({k_label})" if model_ok else "LDA Model", model_state)
        + "</div>"
    )
    st.markdown(pipe_html, unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset Pipeline", use_container_width=True):
        reset_pipeline()
        st.session_state["current_page"] = "🏠 Home Dashboard"
        st.success("Pipeline reset.")
        st.rerun()

    st.markdown(
        '<div class="tl-footer">TopicLens AI &nbsp;·&nbsp; v2.0<br>'
        'Text Topic Modelling &nbsp;·&nbsp; LDA<br>'
        'BCA Capstone &nbsp;·&nbsp; 2024</div>',
        unsafe_allow_html=True,
    )

# ── Router ────────────────────────────────────────────────────────────────────
if   selected_page == "🏠 Home Dashboard":     render_home_view()
elif selected_page == "📂 Dataset Upload":     render_dataset_view()
elif selected_page == "🧹 Data Preprocessing": render_preprocessing_view()
elif selected_page == "⚙️ Topic Modelling":    render_topic_modelling_view()
elif selected_page == "🔍 Topic Exploration":  render_topic_exploration_view()
elif selected_page == "📄 Document Analysis":  render_document_analysis_view()
elif selected_page == "📊 Visual Analytics":   render_visual_analytics_view()
elif selected_page == "📈 Model Evaluation":   render_model_evaluation_view()
elif selected_page == "💾 Export Results":     render_export_results_view()
elif selected_page == "ℹ️ About Project":      render_about_project_view()
