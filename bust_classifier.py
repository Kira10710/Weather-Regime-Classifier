import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score

# --- Load data ---
df = pd.read_csv("corrected_dataset_realistic.csv")
df["valid_date"] = pd.to_datetime(df["valid_date"])
df["month"] = df["valid_date"].dt.month
df["doy"] = df["valid_date"].dt.dayofyear

df["corrected_error_mm"] = df["corrected_precip_mm"] - df["observed_precip_mm"]
df["abs_corrected_error"] = df["corrected_error_mm"].abs()

BUST_THRESHOLD_MM = 6.0
df["is_bust"] = (df["abs_corrected_error"] > BUST_THRESHOLD_MM).astype(int)

print("Bust rate:", df["is_bust"].mean().round(3))

# --- Regime confidence feature (from earlier regime classifier) ---
clf_regime = joblib.load("regime_classifier.pkl")
features_regime = ["forecast_precip_mm", "forecast_temp_c", "month", "doy"]
proba_regime = clf_regime.predict_proba(df[features_regime])
df["regime_confidence"] = proba_regime.max(axis=1)

# --- Train/test split ---
features_bust = ["forecast_precip_mm", "regime_confidence", "month", "doy"]
X = df[features_bust]
y = df["is_bust"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Train Logistic Regression (final chosen model) ---
clf_bust = LogisticRegression(class_weight="balanced", max_iter=1000)
clf_bust.fit(X_train, y_train)

# --- Evaluate at default 0.5 threshold ---
preds_default = clf_bust.predict(X_test)
print("\n--- Default threshold (0.5) ---")
print(classification_report(y_test, preds_default))

# --- Evaluate at chosen threshold (0.6) ---
FINAL_THRESHOLD = 0.6
proba_test = clf_bust.predict_proba(X_test)[:, 1]
preds_final = (proba_test > FINAL_THRESHOLD).astype(int)

print(f"\n--- Final threshold ({FINAL_THRESHOLD}) ---")
print(classification_report(y_test, preds_final))
print(f"Precision: {precision_score(y_test, preds_final):.2f}")
print(f"Recall: {recall_score(y_test, preds_final):.2f}")

# --- Feature influence ---
coefs = pd.Series(clf_bust.coef_[0], index=features_bust)
print("\nFeature influence on bust probability:\n", coefs.sort_values())

# --- Save model + threshold together ---
joblib.dump({"model": clf_bust, "threshold": FINAL_THRESHOLD}, "bust_classifier.pkl")
print("\nSaved bust_classifier.pkl (model + threshold=0.6)")

# --- Apply to FULL dataset for dashboard use ---
proba_full = clf_bust.predict_proba(df[features_bust])[:, 1]
df["bust_probability"] = proba_full
df["is_bust"] = (proba_full > FINAL_THRESHOLD).astype(int)

df.to_csv("final_dataset_with_bust.csv", index=False)
print("Saved final_dataset_with_bust.csv with bust_probability and is_bust (threshold=0.6 applied)")

# OLD (won't work anymore):
# clf_bust = joblib.load("bust_classifier.pkl")

# NEW:
bundle = joblib.load("bust_classifier.pkl")
clf_bust = bundle["model"]
threshold = bundle["threshold"]