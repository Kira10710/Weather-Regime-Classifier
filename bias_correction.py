import pandas as pd

df = pd.read_csv("joined_dataset.csv")
df = df[df["regime"] != "unknown"]

# mean error per regime = how much to shift the forecast
correction = df.groupby("regime")["forecast_error_mm"].mean()
print(correction)

# apply correction: corrected = raw_forecast - mean_error_for_that_regime
df["corrected_precip_mm"] = df.apply(
    lambda r: r["forecast_precip_mm"] - correction[r["regime"]], axis=1
)

# compare RMSE before/after
raw_rmse = ((df["forecast_precip_mm"] - df["observed_precip_mm"])**2).mean()**0.5
corrected_rmse = ((df["corrected_precip_mm"] - df["observed_precip_mm"])**2).mean()**0.5

print(f"Raw RMSE: {raw_rmse:.2f} mm")
print(f"Corrected RMSE: {corrected_rmse:.2f} mm")

df.to_csv("corrected_dataset.csv", index=False)