"""Entry point of the Mini Data Science Laboratory. Run with:  streamlit run app/app.py

This file only sets up the page settings and the list of pages.
Each page lives in its own file in app/views/. The logic is in app/lab.py.

Information is shared between pages with st.session_state:
  st.session_state["dataset"] -> name of the loaded dataset
  st.session_state["df"]      -> the loaded data (pandas DataFrame)
  st.session_state["config"]  -> the experiment configuration
  st.session_state["results"] -> results of the experiments (incl. trained models)
"""
import streamlit as st

st.set_page_config(page_title="Weather Lab", page_icon="🌦️", layout="centered")

home = st.Page("views/home.py", title="Home", icon="🏠", default=True)
load_data = st.Page("views/load_data.py", title="1. Load data", icon="📂")
experiments = st.Page("views/experiments.py", title="2. Experiments", icon="⚙️")
results = st.Page("views/results.py", title="3. Results", icon="📊")
classify = st.Page("views/classify.py", title="4. Predict", icon="🌦️")
about = st.Page("views/about.py", title="About", icon="ℹ️")

nav = st.navigation([home, about, load_data, experiments, results, classify], position="top")
nav.run()
