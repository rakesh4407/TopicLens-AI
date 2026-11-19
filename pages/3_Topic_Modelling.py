"""
pages/3_Topic_Modelling.py
Topic Modelling Training Page
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.session_manager import init_session_state
from src.views.topic_modelling_view import render_topic_modelling_view

init_session_state()
st.session_state['current_page'] = "⚙️ Topic Modelling"
render_topic_modelling_view()
