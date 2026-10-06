"""Starting page: weather photos and two buttons (About / Go to the app)."""
from pathlib import Path

import streamlit as st

IMAGES = Path(__file__).resolve().parents[1] / "images"

st.title("🌦️ Weather Predictor")
st.write(
    "Predict the type of weather from a day's measurements, using k-Nearest Neighbors "
    "trained on two datasets: real Seattle weather and a synthetic weather dataset."
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
    if st.button("🚀 Go to the app", type="primary", width="stretch"):
        st.switch_page("views/choose_k.py")
