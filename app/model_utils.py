"""Shared helpers: dataset settings, loading data and training kNN with a chosen k.

To add a new dataset: put its CSV in data/raw/ and add one entry to DATASETS.
The pages build their inputs from these settings automatically.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "Seattle": {
        "file": "seattle-weather.csv",
        "description": "Real daily weather in Seattle, 2012–2015 (numerical inputs only).",
        "target": "weather",
        "numerical": ["precipitation", "temp_max", "temp_min", "wind"],
        "categorical": [],
        # Nicer names shown in the app (optional)
        "labels": {
            "precipitation": "Precipitation (mm)",
            "temp_max": "Maximum temperature (°C)",
            "temp_min": "Minimum temperature (°C)",
            "wind": "Wind (m/s)",
        },
    },
    "Weather Type": {
        "file": "weather_classification_data.csv",
        "description": "Synthetic (computer-generated) weather data with numerical and categorical inputs.",
        "target": "Weather Type",
        "numerical": ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
                      "Atmospheric Pressure", "UV Index", "Visibility (km)"],
        "categorical": ["Cloud Cover", "Season", "Location"],
        "labels": {
            "Temperature": "Temperature (°C)",
            "Humidity": "Humidity (%)",
            "Wind Speed": "Wind speed (km/h)",
            "Atmospheric Pressure": "Atmospheric pressure (hPa)",
        },
    },
}

# Icons for every class in every dataset
ICONS = {
    "sun": "☀️", "rain": "🌧️", "drizzle": "🌦️", "snow": "❄️", "fog": "🌫️",
    "Sunny": "☀️", "Rainy": "🌧️", "Cloudy": "☁️", "Snowy": "❄️",
}


@st.cache_data
def load_data(name):
    """Read the dataset's CSV once and keep only the columns we use."""
    info = DATASETS[name]
    df = pd.read_csv(DATA / info["file"])
    return df[info["numerical"] + info["categorical"] + [info["target"]]]


@st.cache_resource
def train_model(name, k):
    """Train kNN with k neighbors on one dataset. Cached: each (dataset, k) is trained once.

    Same split as src/main.py: stratified 80/20, random_state=10.
    Returns the model and its accuracy on the test data.
    """
    info = DATASETS[name]
    df = load_data(name)
    X = df[info["numerical"] + info["categorical"]]
    y = df[info["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=10
    )

    # Numbers are scaled; categories become 0/1 columns (one-hot)
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), info["numerical"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), info["categorical"]),
    ])
    model = Pipeline([
        ("prep", preprocess),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)
