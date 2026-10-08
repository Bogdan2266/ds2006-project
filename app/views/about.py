"""Description of the app and info about the authors."""
from pathlib import Path

import streamlit as st

st.title("ℹ️ About this app")

st.header("What does it do?")
st.write(
    "This is a mini data science laboratory. You can load one of two weather datasets and "
    "inspect it, choose how models are evaluated (train/test split or cross-validation, "
    "stratified or not), test several values of k at once, compare accuracy, precision, "
    "recall and F1-score, look at confusion matrices, save the results, and classify new "
    "days with a trained model."
)

st.header("How does it work?")
st.write(
    "The model is **k-Nearest Neighbors (kNN)**. It looks at the *k* days in the training data "
    "that are most similar to your input and picks the most common weather type among them. "
    "Before measuring similarity, numerical features are scaled (StandardScaler) so that no "
    "feature dominates just because of its units, and categorical features (such as season) "
    "are turned into 0/1 columns (one-hot encoding)."
)

st.header("Data")
st.subheader("1. Seattle Weather (real data)")
st.write(
    "Daily weather in Seattle, 2012–2015, from Kaggle. Inputs are numerical only: "
    "precipitation, maximum and minimum temperature, and wind. "
    "Classes: **sun, rain, drizzle, snow, fog**. Because the data comes from one city, "
    "predictions are only meaningful for Seattle-like weather."
)
st.subheader("2. Weather Type Classification (synthetic data)")
st.write(
    "Computer-generated weather data from Kaggle, not tied to a real place. Inputs are "
    "numerical (temperature, humidity, wind speed, precipitation %, pressure, UV index, "
    "visibility) and categorical (cloud cover, season, location type). "
    "Classes: **Sunny, Cloudy, Rainy, Snowy**."
)

st.divider()
st.markdown("<h2 style='text-align:center'>Authors</h2>", unsafe_allow_html=True)

# Photos go in app/images/ (bohdan.jpg, axel.jpg). A placeholder is shown if missing.
IMAGES = Path(__file__).resolve().parents[1] / "images"
AUTHORS = [
    {
        "name": "Bohdan Gertsiuk",
        "photo": "bohdan.jpg",
        "email": "gertsiukbogdan@gmail.com",
        "github": "Bogdan2266",
    },
    {
        "name": "Axel Lundholm",
        "photo": "axel.jpg",
        "email": "axel@example.com",       # change to Axel's email
        "github": "axel-github-username",  # change to Axel's GitHub username
    },
]

columns = st.columns(len(AUTHORS), gap="large")
for column, author in zip(columns, AUTHORS):
    with column:
        st.markdown(f"<h3 style='text-align:center'>{author['name']}</h3>", unsafe_allow_html=True)

        photo = IMAGES / author["photo"]
        if photo.exists():
            st.image(str(photo), width="stretch")
        else:
            st.markdown(
                "<div style='aspect-ratio:1;border:2px dashed #999;border-radius:12px;"
                "display:flex;align-items:center;justify-content:center;font-size:64px'>👤</div>",
                unsafe_allow_html=True,
            )

        github_url = f"https://github.com/{author['github']}"
        st.markdown(
            f"<p style='text-align:center'>✉️ <a href='mailto:{author['email']}'>{author['email']}</a><br>"
            f"🐙 <a href='{github_url}' target='_blank'>github.com/{author['github']}</a></p>",
            unsafe_allow_html=True,
        )

# Info about us (change this text)
st.markdown(
    "<p style='text-align:center;margin-top:24px'>We are students at <b>Halmstad University</b>. "
    "This app is our course project for <b>DS2006 Introduction to Data Science</b>, "
    "where we explore weather data and classify it with k-Nearest Neighbors.</p>",
    unsafe_allow_html=True,
)

st.divider()
if st.button("⬅️ Back to start"):
    st.switch_page("views/home.py")
