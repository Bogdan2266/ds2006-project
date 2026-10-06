"""Weather prediction page: one tab per dataset, using the k chosen before."""
import pandas as pd
import streamlit as st

from model_utils import DATASETS, ICONS, load_data, train_model

k = st.session_state.get("k", 5)  # 5 if the user came here directly

st.title("🌦️ What's the weather like?")
st.info(f"Using kNN with **k = {k}**.")
if st.button("🔢 Change k"):
    st.switch_page("views/choose_k.py")

tabs = st.tabs(list(DATASETS))
for tab, name in zip(tabs, DATASETS):
    with tab:
        info = DATASETS[name]
        df = load_data(name)
        model, accuracy = train_model(name, k)
        st.caption(f"{info['description']} Test accuracy with k = {k}: {accuracy:.1%}")

        # Build one input per column, from the dataset settings
        values = {}
        for column in info["numerical"]:
            label = info["labels"].get(column, column)
            data = df[column]
            # Slider range: 1st–99th percentile (ignores a few extreme values)
            low, high = data.quantile(0.01), data.quantile(0.99)
            if pd.api.types.is_integer_dtype(data):
                values[column] = st.slider(label, int(low), int(high), int(data.median()),
                                           key=f"{name}-{column}")
            else:
                values[column] = st.slider(label, float(round(low)), float(round(high)),
                                           float(round(data.median())), key=f"{name}-{column}")
        for column in info["categorical"]:
            options = sorted(df[column].unique())
            values[column] = st.selectbox(column, options, key=f"{name}-{column}")

        if st.button("Predict", type="primary", key=f"{name}-predict"):
            day = pd.DataFrame([values])  # one row, same column names as in training
            prediction = model.predict(day)[0]
            probabilities = model.predict_proba(day)[0]

            st.success(f"Prediction: {ICONS.get(prediction, '')} **{prediction}**")
            st.write(f"Share of the {k} nearest neighbors in each class:")
            st.bar_chart(pd.Series(probabilities, index=model.classes_))
