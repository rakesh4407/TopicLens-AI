"""
pages/8_Export_Results.py
Export Results & Reports Page
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.session_manager import init_session_state
from src.views.export_results_view import render_export_results_view

init_session_state()
st.session_state['current_page'] = "💾 Export Results"
render_export_results_view()
