"""
app.py
TopicLens AI — Complete NLP Text Topic Modelling Application.
BCA AI & Data Science Capstone Project #9.
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
    has_trained_model
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


# Page configuration
st.set_page_config(
    page_title="TopicLens AI — Text Topic Modelling",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    /* Custom button styling */
    button[kind="primary"] {
        border-radius: 6px;
        font-weight: 500;
    }
    /* Sidebar header */
    .sidebar-brand {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1E3A8A;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .sidebar-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
init_session_state()

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🔍 TopicLens AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Discover Hidden Meaning. Explore Every Topic.<br>Project #9 · BCA AI &amp; Data Science Capstone</div>', unsafe_allow_html=True)

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
        "ℹ️ About Project"
    ]

    # Persistent page selection
    default_index = 0
    if st.session_state.get('current_page') in PAGES:
        default_index = PAGES.index(st.session_state['current_page'])

    selected_page = st.radio(
        "Navigation",
        options=PAGES,
        index=default_index,
        label_visibility="collapsed"
    )
    st.session_state['current_page'] = selected_page

    st.markdown("---")

    # Pipeline Status Box
    st.markdown("##### 📍 Pipeline Status")
    ds_status = "✅ Loaded" if has_dataset() else "❌ None"
    prep_status = "✅ Cleaned" if has_cleaned_data() else "❌ Pending"
    
    tm = st.session_state.get('topic_modeler')
    if tm and tm.is_fitted:
        model_status = f"✅ Fitted (K={tm.n_topics})"
    else:
        model_status = "❌ Not Fitted"

    st.caption(f"**Dataset:** {ds_status}")
    st.caption(f"**Preprocessing:** {prep_status}")
    st.caption(f"**LDA Model:** {model_status}")

    st.markdown("---")

    # Reset Pipeline Action
    if st.button("🔄 Reset Entire Pipeline", use_container_width=True):
        reset_pipeline()
        st.session_state['current_page'] = "🏠 Home Dashboard"
        st.success("Pipeline reset!")
        st.rerun()

    st.caption("Developed for BCA AI & Data Science Capstone Presentation.")


# Route to the appropriate view
if selected_page == "🏠 Home Dashboard":
    render_home_view()
elif selected_page == "📂 Dataset Upload":
    render_dataset_view()
elif selected_page == "🧹 Data Preprocessing":
    render_preprocessing_view()
elif selected_page == "⚙️ Topic Modelling":
    render_topic_modelling_view()
elif selected_page == "🔍 Topic Exploration":
    render_topic_exploration_view()
elif selected_page == "📄 Document Analysis":
    render_document_analysis_view()
elif selected_page == "📊 Visual Analytics":
    render_visual_analytics_view()
elif selected_page == "📈 Model Evaluation":
    render_model_evaluation_view()
elif selected_page == "💾 Export Results":
    render_export_results_view()
elif selected_page == "ℹ️ About Project":
    render_about_project_view()
