"""Description of the app and info about the authors."""
import streamlit as st

st.title("ℹ️ About this app")

st.header("What does it do?")
st.write(
    "You enter the weather measurements of one day and the app predicts the type of weather. "
    "There are two separate models, one for each dataset (see below)."
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
st.write("For both datasets, 80% of the rows are used for training and 20% for testing.")

st.header("Authors")
# Change these lines to your own names
st.write("- **Bogdan Gertsiuk** – Halmstad University")
st.write("- **Axel Lund** – Halmstad University")
st.caption("Course project for DS2006 Introduction to Data Science.")

st.divider()
if st.button("⬅️ Back to start"):
    st.switch_page("views/home.py")
