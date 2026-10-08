"""Step 4: classify new, unseen examples with a trained model (requirement 9)."""
import pandas as pd
import streamlit as st

from lab import DATASETS, check_new_examples, features_of

ICONS = {"sun": "☀️", "rain": "🌧️", "drizzle": "🌦️", "snow": "❄️", "fog": "🌫️",
         "Sunny": "☀️", "Rainy": "🌧️", "Cloudy": "☁️", "Snowy": "❄️"}

st.title("🌦️ What's the weather like?")

if "results" not in st.session_state:
    st.warning("No model has been trained yet. Please configure and run an experiment first.")
    if st.button("Go to 2. Experiments"):
        st.switch_page("views/experiments.py")
    st.stop()

df = st.session_state["df"]
info = DATASETS[st.session_state["dataset"]]
results = st.session_state["results"]

# ---- Choose the trained model ----
# The options are positions in the results list (0, 1, 2, ...); format_func shows "1 - k = 3"
index = st.selectbox("Which trained model should be used?", range(len(results)),
                     format_func=lambda i: f"{results[i]['experiment']} - k = {results[i]['k']}")
chosen = results[index]
model = chosen["model"]
st.info(f"Dataset: **{st.session_state['dataset']}** · Using kNN with **k = {chosen['k']}** "
        f"(F1 = {chosen['f1']:.3f}).")
if st.button("🔢 Change experiments"):
    st.switch_page("views/experiments.py")


def show_prediction(example):
    """Predict one example and show the class with an icon and the neighbor shares."""
    prediction = model.predict(example)[0]
    st.success(f"Prediction: {ICONS.get(prediction, '')} **{prediction}**")
    st.write(f"How confident the model is (share of the {chosen['k']} nearest neighbors in each class):")
    st.bar_chart(pd.Series(model.predict_proba(example)[0], index=model.classes_))


manual_tab, file_tab = st.tabs(["✍️ Enter values", "📄 Load a CSV file"])

# ---- 9.a: manual input ----
with manual_tab:
    values = {}
    for column in info["numerical"]:
        label = info.get("labels", {}).get(column, column)
        data = df[column]
        # Slider range: 1st–99th percentile of the data (ignores a few extreme values)
        low, high = data.quantile(0.01), data.quantile(0.99)
        if pd.api.types.is_integer_dtype(data):
            values[column] = st.slider(label, int(low), int(high), int(data.median()))
        else:
            values[column] = st.slider(label, float(round(low)), float(round(high)),
                                       float(round(data.median())))
    for column in info["categorical"]:
        values[column] = st.selectbox(column, sorted(df[column].unique()))

    if st.button("Predict", type="primary"):
        show_prediction(pd.DataFrame([values]))  # one row, same columns as in training

# ---- 9.b: file input ----
with file_tab:
    st.write("The CSV file must contain these columns: " + ", ".join(f"`{c}`" for c in features_of(info)))
    example_file = df[features_of(info)].head(5).to_csv(index=False)
    st.download_button("Download an example file", data=example_file, file_name="example_input.csv",
                       mime="text/csv")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
        except Exception as error:  # broken or empty file
            st.error(f"Could not read the file: {error}")
            st.stop()

        clean, errors = check_new_examples(new_df, df, info)
        if errors:
            for message in errors:
                st.error(message)
            st.stop()

        output = clean.copy()
        predictions = model.predict(clean)
        output["predicted class"] = [f"{ICONS.get(p, '')} {p}" for p in predictions]
        st.success(f"Classified {len(output)} example(s) with k = {chosen['k']}.")
        st.dataframe(output, hide_index=True)
        st.download_button("💾 Save predictions", data=output.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")
