"""
pages/9_About_Project.py
About Project Information Page
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.session_manager import init_session_state
from src.views.about_project_view import render_about_project_view

init_session_state()
st.session_state['current_page'] = "ℹ️ About Project"
render_about_project_view()
