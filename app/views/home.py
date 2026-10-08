"""Starting page: weather photos and two buttons (About / Start the lab)."""
from pathlib import Path

import streamlit as st

IMAGES = Path(__file__).resolve().parents[1] / "images"

st.title("🌦️ Weather Lab")
st.write(
    "A mini data science laboratory for weather classification with k-Nearest Neighbors. "
    "Load a dataset, design and compare k-NN experiments, save the results, and use a "
    "trained model to classify new days."
)
st.markdown(
    "**Steps:** 📂 1. Load data → ⚙️ 2. Experiments → 📊 3. Results → 🌦️ 4. Predict"
)

# Photos: put your own pictures in app/images/ with these names.
# If a picture is missing, a big emoji is shown instead.
photos = [
    ("sun.jpg", "☀️", "Sun"),
    ("rain.jpg", "🌧️", "Rain"),
    ("snow.jpg", "❄️", "Snow"),
    ("fog.jpg", "🌫️", "Fog"),
]
columns = st.columns(len(photos))
for column, (file_name, emoji, caption) in zip(columns, photos):
    with column:
        path = IMAGES / file_name
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.markdown(
                f"<div style='font-size:64px;text-align:center'>{emoji}</div>"
                f"<p style='text-align:center'>{caption}</p>",
                unsafe_allow_html=True,
            )

st.divider()

left, right = st.columns(2)
with left:
    if st.button("ℹ️ About the app", width="stretch"):
        st.switch_page("views/about.py")
with right:
    if st.button("🚀 Start the lab", type="primary", width="stretch"):
        st.switch_page("views/load_data.py")
