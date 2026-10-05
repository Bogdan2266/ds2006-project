import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_knn(features, k=10):
    """Create a kNN model that prepares numerical and categorical columns automatically."""
    # Find which columns are numbers and which are text (categories)
    numerical = features.select_dtypes(include="number").columns
    categorical = features.select_dtypes(exclude="number").columns

    # Numbers get scaled (kNN measures distances, so all features need similar ranges).
    # Categories get one-hot encoded (e.g. "Winter" -> a column of 0s and 1s).
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def run_knn(csv_path, target, drop_columns=None, k=5, k_values=(1, 3, 5, 7, 9, 11, 15, 21)):
    """Load a dataset, find the best k, train kNN and print the results."""
    print("=" * 60)
    print(f"Dataset: {csv_path}")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    # Remove columns that should not be used as inputs (like 'date')
    if drop_columns:
        df = df.drop(columns=drop_columns)

    # Same idea as in the iris file
    features = df.drop(target, axis=1)
    classes = df[target]

    print("Numerical features:  ", list(features.select_dtypes(include="number").columns))
    print("Categorical features:", list(features.select_dtypes(exclude="number").columns))
    print("Classes:", sorted(classes.unique()))
    print()

    # Stratified split: keeps the same class proportions in the test set
    features_train, features_test, classes_train, classes_test = train_test_split(
        features, classes, test_size=0.2, random_state=10, stratify=classes
    )

    # Try several k values with 5-fold cross-validation on the TRAINING data only.
    # (Choosing k by looking at the test set would be cheating.)
    # --- Best k search (temporarily disabled) ---
    # print("Choosing k (5-fold cross-validation, macro F1):")
    # best_k, best_score = None, -1
    # for k_try in k_values:
    #     model = build_knn(features_train, k_try)
    #     score = cross_val_score(model, features_train, classes_train,
    #                             cv=5, scoring="f1_macro").mean()
    #     print(f"  k = {k_try:2d}  ->  {score:.3f}")
    #     if score > best_score:
    #         best_k, best_score = k_try, score
    # print(f"Best k: {best_k}")
    # print()

    # For now, use the k given when calling run_knn
    best_k = k
    print(f"Using k = {best_k}")
    print()

    # Train the final model with the best k and test it once on the test set
    knn = build_knn(features_train, best_k)
    knn.fit(features_train, classes_train)
    predictions = knn.predict(features_test)

    print("Accuracy:", round(accuracy_score(classes_test, predictions), 3))
    print(classification_report(classes_test, predictions))

    labels = sorted(classes.unique())
    print("Confusion matrix (rows = true class, columns = predicted):")
    print(pd.DataFrame(confusion_matrix(classes_test, predictions, labels=labels),
                       index=labels, columns=labels))
    print()

    return knn


if __name__ == "__main__":
    # Dataset 1: mixed numerical + categorical features
    run_knn("data/raw/weather_type.csv", target="Weather Type")

    # Dataset 2: numerical features only (drop 'date', it is not a feature)
    run_knn("data/raw/seattle_weather.csv", target="weather", drop_columns=["date"])



import joblib
import os

model = run_knn(DATA / "seattle_weather.csv", target="weather", drop_columns=["date"], k=5)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/seattle_knn.joblib")    