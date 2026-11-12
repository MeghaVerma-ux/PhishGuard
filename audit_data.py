
import pandas as pd
import numpy as np
import json
import os

DATA_PATH = "data/final_dataset.csv"
os.makedirs("results", exist_ok=True)

print("Loading complete dataset...")
df = pd.read_csv(DATA_PATH)

numeric = df.select_dtypes(include=np.number)
features = numeric.drop(columns=["label"], errors="ignore")

# 1. Dataset information
report = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "duplicate_rows": int(df.duplicated().sum()),
    "missing_values_total": int(df.isnull().sum().sum()),
    "infinity_values": int(np.isinf(numeric).sum().sum()),
    "target_distribution": {
        str(k): int(v) for k, v in df["label"].value_counts().items()
    },
    "data_types": df.dtypes.astype(str).to_dict(),
    "missing_by_column": {
        k: int(v) for k, v in df.isnull().sum().items() if v > 0
    }
}

# 2. Constant columns
report["constant_columns"] = [
    col for col in features.columns if features[col].nunique(dropna=True) <= 1
]

# 3. IQR outlier analysis
outliers = {}
for col in features.columns:
    series = features[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) == 0:
        continue
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    if count:
        outliers[col] = count

report["outlier_counts_by_column"] = outliers

# 4. Save reports
with open("results/preprocessing_audit.json", "w") as f:
    json.dump(report, f, indent=4)

pd.DataFrame({
    "column": features.columns,
    "missing": features.isnull().sum().values,
    "unique_values": features.nunique().values
}).to_csv("results/feature_audit.csv", index=False)

print("\n========== DATASET AUDIT ==========")
print("Rows:", report["rows"])
print("Columns:", report["columns"])
print("Duplicate rows:", report["duplicate_rows"])
print("Missing values:", report["missing_values_total"])
print("Infinite values:", report["infinity_values"])
print("Constant columns:", report["constant_columns"])
print("\nTarget distribution:")
print(df["label"].value_counts())
print("\nTop 10 columns with outliers:")
print(dict(sorted(outliers.items(), key=lambda x: x[1], reverse=True)[:10]))
print("\nReports saved in results/ folder.")
print("===================================")
