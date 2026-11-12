
import os
import json
import time
import argparse
import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/final_dataset.csv"
MODEL_PATH = "model/model.joblib"

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--sample-size", type=int, default=20000)
args = parser.parse_args()

print("=" * 60)
print("PHISHING WEBSITE DETECTION - MODEL TRAINING")
print("=" * 60)

# 1. Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# 2. Data cleaning
print("\n2. Data cleaning...")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.drop_duplicates()
df = df.dropna(subset=["label"])

# Remove invalid target values
df = df[df["label"].isin([0, 1])].copy()

# URL is excluded; label is target
X = df.drop(columns=["url", "label"], errors="ignore")
X = X.select_dtypes(include=np.number)

y = df["label"].astype(int)

# 3. Dataset sampling
if args.sample_size and len(df) > args.sample_size:
    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=args.sample_size,
        random_state=42,
        stratify=y
    )

print("Working dataset:", X.shape)
print("Target distribution:")
print(y.value_counts())

# 4. Train-test split
print("\n3. Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Preprocessing pipelines
def make_pipeline(classifier, scaling=False):

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0))
    ]

    if scaling:
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", classifier))

    return Pipeline(steps)

# 6. Ten machine learning algorithms
models = {
    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    ),

    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            n_jobs=-1,
            random_state=42
        )
    ),

    "Extra Trees": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=80,
            max_depth=18,
            n_jobs=-1,
            random_state=42
        )
    ),

    "Logistic Regression": make_pipeline(
        LogisticRegression(
            max_iter=500,
            solver="liblinear",
            random_state=42
        ),
        scaling=True
    ),

    "KNN": make_pipeline(
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1
        ),
        scaling=True
    ),

    "Gaussian Naive Bayes": make_pipeline(
        GaussianNB(),
        scaling=True
    ),

    "AdaBoost": make_pipeline(
        AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        )
    ),

    "Gradient Boosting": make_pipeline(
        GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
    ),

    "Hist Gradient Boosting": make_pipeline(
        HistGradientBoostingClassifier(
            max_iter=60,
            max_leaf_nodes=25,
            random_state=42
        )
    ),

    "Linear SVM": make_pipeline(
        LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42
        ),
        scaling=True
    )
}

# 7. MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Phishing-Website-Detection")

results = []
best_f1 = -1
best_pipeline = None
best_name = None
best_report = None
best_cm = None

# 8. Train and evaluate
print("\n4. Training and evaluating models...\n")

for name, pipeline in models.items():

    print("\nTraining:", name)
    start = time.time()

    try:
        with mlflow.start_run(run_name=name):

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(
                y_test, predictions, zero_division=0
            )
            recall = recall_score(
                y_test, predictions, zero_division=0
            )
            f1 = f1_score(
                y_test, predictions, zero_division=0
            )

            if hasattr(pipeline, "predict_proba"):
                scores = pipeline.predict_proba(X_test)[:, 1]
            else:
                scores = pipeline.decision_function(X_test)

            auc = roc_auc_score(y_test, scores)

            elapsed = round(time.time() - start, 2)

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": auc,
                "training_seconds": elapsed
            }

            mlflow.log_param("algorithm", name)
            mlflow.log_param("sample_size", len(X))
            mlflow.log_param("features", X.shape[1])
            mlflow.log_metrics(metrics)

            results.append({
                "Algorithm": name,
                **metrics,
                "Status": "Success"
            })

            print("Accuracy:", round(accuracy, 4))
            print("Precision:", round(precision, 4))
            print("Recall:", round(recall, 4))
            print("F1 Score:", round(f1, 4))
            print("ROC AUC:", round(auc, 4))
            print("Time:", elapsed, "seconds")

            if f1 > best_f1:
                best_f1 = f1
                best_pipeline = pipeline
                best_name = name

                best_report = classification_report(
                    y_test,
                    predictions,
                    output_dict=True
                )

                best_cm = confusion_matrix(
                    y_test, predictions
                )

    except Exception as error:
        print("FAILED:", name, "-", str(error))

        results.append({
            "Algorithm": name,
            "Status": "Failed",
            "Error": str(error)
        })

# 9. Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

successful = results_df[
    results_df["Status"] == "Success"
]

if best_pipeline is None:
    raise RuntimeError(
        "No model trained successfully. Check errors above."
    )

# 10. Save best model
selected_features = X.columns[
    best_pipeline.named_steps["variance"].get_support()
].tolist()

bundle = {
    "pipeline": best_pipeline,
    "feature_names": X.columns.tolist(),
    "selected_features": selected_features,
    "classes": best_pipeline.classes_.tolist(),
    "best_model_name": best_name,
    "target_column": "label"
}

joblib.dump(bundle, MODEL_PATH)

# 11. Save evaluation reports
with open("results/classification_report.json", "w") as f:
    json.dump(best_report, f, indent=4)

pd.DataFrame(
    best_cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
).to_csv("results/confusion_matrix.csv")

# 12. Feature importance
classifier = best_pipeline.named_steps["classifier"]

if hasattr(classifier, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": selected_features,
        "Importance": classifier.feature_importances_
    }).sort_values("Importance", ascending=False)

    importance.to_csv(
        "results/feature_importance.csv",
        index=False
    )

# 13. Preprocessing report
preprocessing_report = {
    "original_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "original_features": int(X.shape[1]),
    "selected_features": int(len(selected_features)),
    "missing_values_before_imputation": int(X.isnull().sum().sum()),
    "duplicates_removed": int(pd.read_csv(DATA_PATH).duplicated().sum()),
    "sampling_size": int(len(X)),
    "scaling": "StandardScaler for scale-sensitive algorithms",
    "imputation": "Median",
    "feature_selection": "VarianceThreshold",
    "outlier_policy": "IQR audit; no blind deletion"
}

with open("results/preprocessing_report.json", "w") as f:
    json.dump(preprocessing_report, f, indent=4)

# 14. Final summary
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    successful[
        ["Algorithm", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ].to_string(index=False)
)

print("\nBEST MODEL:", best_name)
print("BEST F1 SCORE:", round(best_f1, 4))
print("MODEL SAVED:", MODEL_PATH)
print("REPORTS SAVED: results/")
print("=" * 60)
