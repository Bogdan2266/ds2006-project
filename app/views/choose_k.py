"""Let the user choose k (number of neighbors) before predicting."""
import streamlit as st

from model_utils import DATASETS, train_model

st.title("🔢 Choose k")
st.write(
    "**k** is the number of neighbors (most similar days) the model looks at. "
    "Small k follows the training data very closely (can overfit); "
    "large k gives smoother, more general answers (can underfit)."
    "According to our tests, the best k is around 5–10 for both datasets."
    "We recommend trying a few values and seeing how the accuracy changes."
    "The best results are obtained with k=5 for the Seattle dataset and k=10 for the weather prediction dataset."
)

# Remember the choice between pages with st.session_state
k = st.slider("Number of neighbors (k)", min_value=1, max_value=50,
              value=st.session_state.get("k", 5))

st.write("Accuracy on the test data with this k:")
columns = st.columns(len(DATASETS))
for column, name in zip(columns, DATASETS):
    model, accuracy = train_model(name, k)
    column.metric(name, f"{accuracy:.1%}")

st.divider()
left, right = st.columns(2)
with left:
    if st.button("⬅️ Back", width="stretch"):
        st.switch_page("views/home.py")
with right:
    if st.button("Continue ➡️", type="primary", width="stretch"):
        st.session_state["k"] = k
        st.switch_page("views/predict.py")
