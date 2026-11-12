
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
DATA_PATH = BASE_DIR / "data" / "final_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

print("Loading trained model...")

bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]
feature_names = list(bundle["feature_names"])
model_name = bundle["best_model_name"]

print("Model:", model_name)
print("Features:", len(feature_names))

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=feature_names + ["label"]
)

# Take 500 records from each class
sample = (
    df.groupby("label", group_keys=False)
      .sample(n=500, random_state=42)
      .reset_index(drop=True)
)

X = sample[feature_names].astype(float)
y = sample["label"].astype(int)

print("\nSample distribution:")
print(y.value_counts())

print("\nCalculating permutation importance...")
print("Please wait...")

importance = permutation_importance(
    pipeline,
    X,
    y,
    scoring="f1",
    n_repeats=3,
    random_state=42,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance.importances_mean,
    "Standard_Deviation": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

csv_path = RESULTS_DIR / "feature_importance.csv"

importance_df.to_csv(csv_path, index=False)

# Plot top 20 features
top_features = importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Permutation Importance (F1 score decrease)")
plt.ylabel("Features")
plt.title("Top 20 Features - PhishGuard")

plt.tight_layout()

plot_path = RESULTS_DIR / "feature_importance.png"

plt.savefig(plot_path, dpi=300)
plt.close()

print("\nTOP 20 IMPORTANT FEATURES:")
print(importance_df.head(20).to_string(index=False))

print("\nFiles saved successfully:")
print(csv_path)
print(plot_path)

print("\nFeature importance analysis completed!")
