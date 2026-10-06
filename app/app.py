from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weather Predictor", page_icon="🌦️")

# Load the trained model (created by running main.py)
MODEL_PATH = Path(__file__).parent.parent / "models" / "seattle_knn.joblib"
model = joblib.load(MODEL_PATH)

st.title("🌦️ What's the weather like?")
st.write("Enter the day's measurements and the kNN model will predict the weather type.")

precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
temp_max = st.slider("Maximum temperature (°C)", -10.0, 40.0, 15.0)
temp_min = st.slider("Minimum temperature (°C)", -15.0, 25.0, 7.0)
wind = st.slider("Wind (m/s)", 0.0, 10.0, 3.0)

if st.button("Predict"):
    # One row with the same column names the model was trained on
    day = pd.DataFrame([{
        "precipitation": precipitation,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "wind": wind,
    }])

    prediction = model.predict(day)[0]
    probabilities = model.predict_proba(day)[0]

    icons = {"sun": "☀️", "rain": "🌧️", "drizzle": "🌦️", "snow": "❄️", "fog": "🌫️"}
    st.success(f"Prediction: {icons.get(prediction, '')} **{prediction}**")

    st.write("How confident the model is (share of neighbors in each class):")
    st.bar_chart(pd.Series(probabilities, index=model.classes_))