"""Step 1: choose, load and inspect a dataset (requirement 2)."""
import streamlit as st

from lab import DATASETS, feature_types, features_of, load_dataset

st.title("📂 1. Load a dataset")

name = st.selectbox("Choose a dataset", list(DATASETS))
st.caption(DATASETS[name]["description"])

if st.button("Load dataset", type="primary"):
    try:
        df = load_dataset(name)
    except (FileNotFoundError, KeyError) as error:
        st.error(f"Could not load the dataset: {error}")
    else:
        # A new dataset makes old experiments invalid, so we remove them
        if st.session_state.get("dataset") != name:
            st.session_state.pop("config", None)
            st.session_state.pop("results", None)
        st.session_state["dataset"] = name
        st.session_state["df"] = df

if "df" not in st.session_state:
    st.info("No dataset loaded yet. Choose one above and press **Load dataset**.")
    st.stop()  # nothing more to show on this page

# ---- Information about the loaded dataset ----
df = st.session_state["df"]
info = DATASETS[st.session_state["dataset"]]
target = info["target"]

st.success(f"✅ Dataset **{st.session_state['dataset']}** loaded correctly.")

st.subheader("First 10 rows")
st.dataframe(df.head(10))

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Classes", df[target].nunique())

st.subheader("Features")
st.write(f"**Input features ({len(features_of(info))}):** {', '.join(features_of(info))}")
st.write(f"**Target (class) column:** {target}")
st.dataframe(feature_types(df, info), hide_index=True)
if info["categorical"]:
    st.caption("Categorical features will be converted to numbers with one-hot encoding before training.")
else:
    st.caption("All features are numerical, so they can be used by k-NN directly (after scaling).")

st.subheader("Target classes")
counts = df[target].value_counts()
class_table = counts.rename("instances").to_frame()
class_table["share"] = (counts / counts.sum()).map("{:.1%}".format)
st.dataframe(class_table)
st.bar_chart(counts)

st.subheader("Descriptive statistics (numerical features)")
st.dataframe(df[info["numerical"]].describe().round(2))

if info["categorical"]:
    st.subheader("Categorical features")
    for column in info["categorical"]:
        st.write(f"**{column}:** " + ", ".join(f"{v} ({c})" for v, c in df[column].value_counts().items()))

st.divider()
if st.button("Next: configure experiments ➡️", type="primary"):
    st.switch_page("views/experiments.py")
