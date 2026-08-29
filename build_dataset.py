"""
Joins forecast (GFS) + observed (IMERG) + regime label into ONE table.
This is the single most important artifact of the whole project -
every model in Day 2/3 trains on this file.

BEFORE RUNNING:
    1. Run fetch_gfs.py and fetch_imerg.py first.
    2. Create regime_labels.csv by hand (~20 min task) with two columns:
           date,regime
           2023-06-15,active
           2023-06-20,break
           ...
       Source: IMD's public monsoon bulletins (search "IMD active break
       monsoon bulletin <year>"). You only need enough dates to label
       every day in your range - active/break periods are usually
       announced as date RANGES, so expand each range to one row per day.

OUTPUT COLUMNS (final joined_dataset.csv):
    valid_date, region, forecast_precip_mm, forecast_temp_c,
    observed_precip_mm, regime, forecast_error_mm
"""

import pandas as pd
import config as cfg


def main():
    gfs = pd.read_csv(cfg.GFS_OUT_CSV)
    imerg = pd.read_csv(cfg.IMERG_OUT_CSV)

    df = gfs.merge(imerg, on=["valid_date", "region"], how="inner")

    try:
        regimes = pd.read_csv(cfg.REGIME_LABELS_CSV)
        regimes = regimes.rename(columns={"date": "valid_date"})
        df = df.merge(regimes, on="valid_date", how="left")
        df["regime"] = df["regime"].fillna("unknown")
    except FileNotFoundError:
        print(f"[warning] {cfg.REGIME_LABELS_CSV} not found - "
              f"filling regime with 'unknown'. Create this file by hand "
              f"before Day 2 (see docstring above).")
        df["regime"] = "unknown"

    # ADD THIS LINE HERE — fix duplicate region columns from the merge
    if "region_x" in df.columns:
        df = df.drop(columns=["region_y"]).rename(columns={"region_x": "region"})

    df["forecast_error_mm"] = df["forecast_precip_mm"] - df["observed_precip_mm"]

    df.to_csv(cfg.JOINED_OUT_CSV, index=False)
    print(f"Saved {len(df)} joined rows to {cfg.JOINED_OUT_CSV}")
    print("\nPreview:")
    print(df.head())

    print("\nRegime counts:")
    print(df["regime"].value_counts())

    print(f"\nRaw forecast RMSE: "
          f"{(df['forecast_error_mm'] ** 2).mean() ** 0.5:.2f} mm")


if __name__ == "__main__":
    main()