"""Step 2: choose the evaluation strategy and the k values (requirements 4 and 5)."""
import streamlit as st

from lab import (DATASETS, describe_config, run_experiments, training_size,
                 validate_folds, validate_k_values, validate_split)

st.title("⚙️ 2. Configure the experiments")

if "df" not in st.session_state:
    st.warning("Please load a dataset first.")
    if st.button("Go to 1. Load data"):
        st.switch_page("views/load_data.py")
    st.stop()

df = st.session_state["df"]
info = DATASETS[st.session_state["dataset"]]
y = df[info["target"]]
st.write(f"Dataset: **{st.session_state['dataset']}** ({len(df)} rows)")

errors = []  # every validation problem is collected here
config = {"dataset": st.session_state["dataset"]}

# ---- Data partitioning strategy ----
st.subheader("Data partitioning strategy")
strategy = st.radio("How should the models be evaluated?",
                    ["Train/test split", "X-fold cross-validation"], horizontal=True)

if strategy == "Train/test split":
    config["strategy"] = "split"
    col1, col2 = st.columns(2)
    config["train_pct"] = int(col1.number_input("Training percentage (%)", value=70, step=1))
    config["test_pct"] = int(col2.number_input("Testing percentage (%)", value=30, step=1))
    errors += validate_split(config["train_pct"], config["test_pct"])
else:
    config["strategy"] = "cv"
    config["folds"] = int(st.number_input("Number of folds", value=5, step=1))
config["stratified"] = st.toggle("Stratified (keep the same class proportions in every part)", value=True)
if config["strategy"] == "cv":
    errors += validate_folds(config["folds"], y, config["stratified"])

# ---- k-NN experiments ----
st.subheader("k-NN experiments")
n_experiments = int(st.number_input("How many k-NN variations would you like to test?",
                                    min_value=1, max_value=10, value=3, step=1))
default_k = [3, 5, 9, 15, 21, 31, 41, 51, 75, 101]
columns = st.columns(min(n_experiments, 5))
k_values = []
for i in range(n_experiments):
    column = columns[i % len(columns)]
    k_values.append(int(column.number_input(f"k for experiment {i + 1}", value=default_k[i],
                                            step=1, key=f"k_{i}")))
config["k_values"] = k_values

if not errors:  # training size can only be computed with valid strategy settings
    errors += validate_k_values(k_values, training_size(len(df), config))

# ---- Show configuration or errors ----
st.divider()
if errors:
    for message in errors:
        st.error(message)
    st.info("Fix the problems above to run the experiments.")
    st.stop()

st.subheader("Experiment configuration")
st.code(describe_config(config), language=None)

if st.button("▶️ Run experiments", type="primary"):
    with st.spinner("Training and evaluating the models..."):
        st.session_state["results"] = run_experiments(df, info, config)
        st.session_state["config"] = config
    st.switch_page("views/results.py")
