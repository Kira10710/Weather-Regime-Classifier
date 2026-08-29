import pandas as pd
import joblib

df = pd.read_csv("joined_dataset.csv")
df = df[df["regime"] != "unknown"].copy()

df["valid_date"] = pd.to_datetime(df["valid_date"])
df["month"] = df["valid_date"].dt.month
df["doy"] = df["valid_date"].dt.dayofyear

clf = joblib.load("regime_classifier.pkl")

features = ["forecast_precip_mm", "forecast_temp_c", "month", "doy"]
df["predicted_regime"] = clf.predict(df[features])

# correction offsets learned earlier (recompute from true labels, as before)
correction = df.groupby("regime")["forecast_error_mm"].mean()

# apply correction using PREDICTED regime, not true regime
df["corrected_precip_mm"] = df.apply(
    lambda r: r["forecast_precip_mm"] - correction[r["predicted_regime"]], axis=1
)

raw_rmse = ((df["forecast_precip_mm"] - df["observed_precip_mm"])**2).mean()**0.5
corrected_rmse_true_regime = ((df["forecast_precip_mm"] - correction[df["regime"]].values - df["observed_precip_mm"])**2).mean()**0.5
corrected_rmse_predicted_regime = ((df["corrected_precip_mm"] - df["observed_precip_mm"])**2).mean()**0.5

print(f"Raw RMSE:                          {raw_rmse:.2f} mm")
print(f"Corrected RMSE (true regime):       {corrected_rmse_true_regime:.2f} mm")
print(f"Corrected RMSE (predicted regime):  {corrected_rmse_predicted_regime:.2f} mm")

df.to_csv("corrected_dataset_realistic.csv", index=False)