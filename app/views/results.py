"""Step 3: metrics, confusion matrix and saving the results (requirements 6, 7 and 8)."""
import streamlit as st

from lab import clean_filename, describe_config, results_table

st.title("📊 3. Results")

if "results" not in st.session_state:
    st.warning("No experiments have been run yet. Configure and run them first.")
    if st.button("Go to 2. Experiments"):
        st.switch_page("views/experiments.py")
    st.stop()

config = st.session_state["config"]
results = st.session_state["results"]
table = results_table(config, results)

with st.expander("Experiment configuration", expanded=False):
    st.code(describe_config(config), language=None)

# ---- Metrics (requirement 6) ----
st.subheader("Metrics for every k")
st.caption("Precision, recall and F1 use the **macro** average: the score is computed for each "
           "class and then averaged, so every class counts equally.")
metrics = ["accuracy", "precision (macro)", "recall (macro)", "f1 (macro)"]
st.dataframe(table[["experiment", "k"] + metrics], hide_index=True)

best = table.loc[table["f1 (macro)"].idxmax()]
st.success(f"Best F1-score: **k = {best['k']}** (F1 = {best['f1 (macro)']:.3f})")

chart_data = table.set_index("k")[metrics]
chart_data.index = chart_data.index.map(lambda k: f"k = {k}")
st.bar_chart(chart_data, stack=False)

# ---- Confusion matrix (requirement 7) ----
st.subheader("Confusion matrix")
# The options are positions in the results list (0, 1, 2, ...); format_func shows "1 - k = 3"
index = st.selectbox("Which experiment would you like to inspect?", range(len(results)),
                     format_func=lambda i: f"{results[i]['experiment']} - k = {results[i]['k']}")
st.dataframe(results[index]["confusion_matrix"])
st.caption("Rows = actual class, columns = predicted class. Numbers on the diagonal are correct predictions.")

# ---- Save results (requirement 8) ----
st.subheader("Save the results")
typed_name = st.text_input("File name", value="knn_results")
file_name = clean_filename(typed_name)
if file_name is None:
    st.error("Please enter a valid file name (letters, numbers, - or _).")
else:
    st.download_button(f"💾 Save as {file_name}", data=table.to_csv(index=False),
                       file_name=file_name, mime="text/csv")

st.divider()
if st.button("Next: predict new examples ➡️", type="primary"):
    st.switch_page("views/classify.py")
