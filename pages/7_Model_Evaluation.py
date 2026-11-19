"""
pages/7_Model_Evaluation.py
Model Evaluation Page
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.session_manager import init_session_state
from src.views.model_evaluation_view import render_model_evaluation_view

init_session_state()
st.session_state['current_page'] = "📈 Model Evaluation"
render_model_evaluation_view()
