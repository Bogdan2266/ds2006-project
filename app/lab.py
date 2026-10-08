"""Core logic of the Mini Data Science Laboratory.

This file has NO Streamlit code: only pandas and scikit-learn.
The pages in app/views/ call these functions and show the results.
Keeping the logic here makes it easy to test and to explain.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "raw"
RANDOM_STATE = 10  # same random_state everywhere, so results can be repeated

# ---------------------------------------------------------------------------
# 1. Datasets supported by the laboratory
# ---------------------------------------------------------------------------
DATASETS = {
    "Seattle Weather (numerical)": {
        "file": "seattle-weather.csv",
        "description": "Real daily weather in Seattle, 2012–2015. Numerical inputs only.",
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
    "Weather Type (mixed)": {
        "file": "weather_classification_data.csv",
        "description": "Synthetic weather data with numerical AND categorical inputs.",
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


def features_of(info):
    """All input feature names of a dataset (numerical first, then categorical)."""
    return info["numerical"] + info["categorical"]


def load_dataset(name):
    """Read the CSV of a dataset and keep only the columns we use.

    Raises FileNotFoundError / KeyError with a clear message if something is wrong.
    """
    info = DATASETS[name]
    path = DATA / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: data/raw/{info['file']}")
    df = pd.read_csv(path)
    missing = [c for c in features_of(info) + [info["target"]] if c not in df.columns]
    if missing:
        raise KeyError(f"The file is missing these columns: {missing}")
    return df[features_of(info) + [info["target"]]]


def feature_types(df, info):
    """Table that tells the user which features are numerical and which are categorical."""
    rows = []
    for column in features_of(info):
        kind = "categorical" if column in info["categorical"] else "numerical"
        rows.append({"feature": column, "type": kind, "pandas dtype": str(df[column].dtype),
                     "unique values": df[column].nunique()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Data preparation + model
# ---------------------------------------------------------------------------
def build_model(info, k):
    """Pipeline = data preparation + kNN.

    - numerical columns  -> StandardScaler (same scale for all features)
    - categorical columns -> OneHotEncoder (text categories become 0/1 columns)
    Because preparation is INSIDE the pipeline, new examples get exactly
    the same preparation as the training data.
    """
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), info["numerical"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), info["categorical"]),
    ])
    return Pipeline([
        ("prepare", preprocess),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


# ---------------------------------------------------------------------------
# 3. Validation of user input. Each function returns a list of error messages
#    (an empty list means everything is OK).
# ---------------------------------------------------------------------------
def validate_split(train_pct, test_pct):
    errors = []
    if train_pct + test_pct != 100:
        errors.append(f"Training + testing must be 100% (now {train_pct} + {test_pct} = {train_pct + test_pct}%).")
    if train_pct <= 0 or test_pct <= 0:
        errors.append("Both training and testing percentages must be greater than 0.")
    return errors


def validate_folds(folds, y, stratified):
    errors = []
    if folds < 2:
        errors.append("Cross-validation needs at least 2 folds.")
    if folds > len(y):
        errors.append(f"Number of folds cannot be larger than the number of rows ({len(y)}).")
    smallest_class = y.value_counts().min()
    if stratified and folds > smallest_class:
        errors.append(f"With stratification, folds cannot be more than the smallest class size ({smallest_class}).")
    return errors


def training_size(n_rows, config):
    """How many rows the model is trained on (k cannot be larger than this)."""
    if config["strategy"] == "split":
        return int(n_rows * config["train_pct"] / 100)
    return n_rows - n_rows // config["folds"] - 1  # smallest training part in CV


def validate_k_values(k_values, max_k):
    errors = []
    if len(k_values) == 0:
        errors.append("Add at least one experiment.")
    for i, k in enumerate(k_values, start=1):
        if k < 1:
            errors.append(f"Experiment {i}: k must be at least 1 (got {k}).")
        elif k > max_k:
            errors.append(f"Experiment {i}: k = {k} is larger than the training data ({max_k} rows).")
    if len(set(k_values)) != len(k_values):
        errors.append("Each experiment must use a different k (some values are repeated).")
    return errors


def describe_config(config):
    """Human-readable text of the experiment configuration."""
    stratified = "Stratified" if config["stratified"] else "Non-stratified"
    if config["strategy"] == "split":
        strategy = f"{stratified} train/test split: {config['train_pct']}% training, {config['test_pct']}% testing"
    else:
        strategy = f"{config['folds']}-fold {stratified} cross-validation"
    lines = [f"Dataset: {config['dataset']}", f"Data partitioning strategy: {strategy}", "k-NN experiments:"]
    for i, k in enumerate(config["k_values"], start=1):
        lines.append(f"  Experiment {i}: k = {k}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Run the experiments
# ---------------------------------------------------------------------------
def run_experiments(df, info, config):
    """Train and evaluate one kNN model per k, all with the SAME evaluation setup.

    Returns a list with one dict per experiment: k, metrics, confusion matrix,
    and a trained model that can classify new examples.
    """
    X = df[features_of(info)]
    y = df[info["target"]]
    labels = sorted(y.unique())  # all classes, so every confusion matrix has the same shape
    stratified = config["stratified"]

    if config["strategy"] == "split":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_pct"] / 100,
            stratify=y if stratified else None,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    else:
        if stratified:
            splitter = StratifiedKFold(n_splits=config["folds"], shuffle=True, random_state=RANDOM_STATE)
        else:
            splitter = KFold(n_splits=config["folds"], shuffle=True, random_state=RANDOM_STATE)

    results = []
    for number, k in enumerate(config["k_values"], start=1):
        if config["strategy"] == "split":
            model = build_model(info, k)
            model.fit(X_train, y_train)
            y_true, y_pred = y_test, model.predict(X_test)
        else:
            # Every row is predicted once, by a model that did not see it in training
            y_true = y
            y_pred = cross_val_predict(build_model(info, k), X, y, cv=splitter)
            # Final model for classifying new examples: trained on ALL rows
            model = build_model(info, k)
            model.fit(X, y)

        results.append({
            "experiment": number,
            "k": k,
            # Multiclass: "macro" average = mean of the per-class scores (every class counts equally)
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "confusion_matrix": pd.DataFrame(
                confusion_matrix(y_true, y_pred, labels=labels),
                index=[f"actual: {c}" for c in labels],
                columns=[f"predicted: {c}" for c in labels],
            ),
            "model": model,
        })
    return results


def results_table(config, results):
    """One row per experiment with configuration + metrics (used on screen and for saving)."""
    rows = []
    for r in results:
        rows.append({
            "dataset": config["dataset"],
            "strategy": "train/test split" if config["strategy"] == "split" else "cross-validation",
            "train %": config.get("train_pct") if config["strategy"] == "split" else None,
            "test %": config.get("test_pct") if config["strategy"] == "split" else None,
            "folds": config.get("folds") if config["strategy"] == "cv" else None,
            "stratified": config["stratified"],
            "experiment": r["experiment"],
            "k": r["k"],
            "accuracy": round(r["accuracy"], 4),
            "precision (macro)": round(r["precision"], 4),
            "recall (macro)": round(r["recall"], 4),
            "f1 (macro)": round(r["f1"], 4),
        })
    return pd.DataFrame(rows)


def clean_filename(name):
    """Make a safe CSV file name from what the user typed. Returns None if it is empty."""
    name = name.strip()
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_ .").strip()
    if not safe:
        return None
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    return safe


# ---------------------------------------------------------------------------
# 5. Classify new, unseen examples
# ---------------------------------------------------------------------------
def check_new_examples(new_df, df, info):
    """Check a table of new examples before classifying. Returns (clean_table, errors)."""
    errors = []
    missing = [c for c in features_of(info) if c not in new_df.columns]
    if missing:
        return None, [f"Missing columns: {missing}. Needed: {features_of(info)}"]
    if len(new_df) == 0:
        return None, ["The file has no rows."]

    clean = new_df[features_of(info)].copy()
    for column in info["numerical"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if clean[column].isna().any():
            errors.append(f"Column '{column}' must contain numbers only (empty or text values found).")
    for column in info["categorical"]:
        known = set(df[column].unique())
        unknown = set(clean[column].astype(str)) - known
        if unknown:
            errors.append(f"Column '{column}' has unknown values {sorted(unknown)}. Allowed: {sorted(known)}")
    return clean, errors
