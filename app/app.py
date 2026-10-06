"""Entry point of the website. Run with:  streamlit run app/app.py

This file only sets up the page settings and the list of pages.
Each page lives in its own file in the app/views/ folder.
"""
import streamlit as st

st.set_page_config(page_title="Weather Predictor", page_icon="🌦️", layout="centered")

# All pages of the website (file, title in the menu, icon)
home = st.Page("views/home.py", title="Home", icon="🏠", default=True)
about = st.Page("views/about.py", title="About", icon="ℹ️")
choose_k = st.Page("views/choose_k.py", title="Choose k", icon="🔢")
predict = st.Page("views/predict.py", title="Predict", icon="🌦️")

# Menu at the top of the page; switch_page() in the pages moves between them1
nav = st.navigation([home, about, choose_k, predict], position="top")
nav.run()
