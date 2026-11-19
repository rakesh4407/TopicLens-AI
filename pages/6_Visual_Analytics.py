"""
pages/6_Visual_Analytics.py
Visual Analytics Page
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.session_manager import init_session_state
from src.views.visual_analytics_view import render_visual_analytics_view

init_session_state()
st.session_state['current_page'] = "📊 Visual Analytics"
render_visual_analytics_view()
