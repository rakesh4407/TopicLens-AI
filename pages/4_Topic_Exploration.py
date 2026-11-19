"""
pages/4_Topic_Exploration.py
Topic Exploration Page
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.session_manager import init_session_state
from src.views.topic_exploration_view import render_topic_exploration_view

init_session_state()
st.session_state['current_page'] = "🔍 Topic Exploration"
render_topic_exploration_view()
