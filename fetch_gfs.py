"""
Fetch historical GFS forecast data for your region/date-range/lead-time.

SETUP (run once, on your own machine):
    pip install herbie-data pandas xarray cfgrib

WHAT THIS DOES:
    For each day in your date range, downloads the GFS 0.25-degree forecast
    (run at RUN_HOUR, valid at +LEAD_HOURS), subsets it to your region's
    bounding box, and averages APCP (precip) and TMP (temperature) over
    that box. Appends one row per day to a CSV.

WHY HERBIE:
    Herbie finds the right file on NOAA/AWS/Google archives automatically
    and only downloads the bytes for the variables you ask for (via GRIB
    index/byte-range under the hood) instead of the whole global file.

NOTE:
    This will NOT run in a sandboxed/offline environment - it needs real
    internet access to NOAA/AWS servers. Run it on your own laptop.
"""

import pandas as pd
from datetime import datetime, timedelta
from herbie import Herbie

import config as cfg


def fetch_one_day(date_str: str) -> dict | None:
    """Fetch GFS forecast for a single init date, return region-averaged values."""
    try:
        H = Herbie(
            date_str,
            model="gfs",
            product="pgrb2.0p25",
            fxx=cfg.LEAD_HOURS,
        )

        # Pull precipitation and temperature, subset to bounding box
        ds_precip = H.xarray(":APCP:surface", remove_grib=True)
        ds_temp = H.xarray(":TMP:2 m above ground", remove_grib=True)

        def region_mean(ds):
            # GFS longitudes are 0-360, convert bounding box if needed
            lon_min = cfg.LON_MIN % 360
            lon_max = cfg.LON_MAX % 360
            box = ds.sel(
                latitude=slice(cfg.LAT_MAX, cfg.LAT_MIN),  # lat descends in GFS grids
                longitude=slice(lon_min, lon_max),
            )
            return float(box[list(box.data_vars)[0]].mean().values)

        precip_mm = region_mean(ds_precip)
        temp_k = region_mean(ds_temp)

        return {
            "init_date": date_str,
            "valid_date": (
                datetime.strptime(date_str, "%Y-%m-%d")
                + timedelta(hours=cfg.LEAD_HOURS)
            ).strftime("%Y-%m-%d"),
            "lead_hours": cfg.LEAD_HOURS,
            "region": cfg.REGION_NAME,
            "forecast_precip_mm": precip_mm,
            "forecast_temp_c": temp_k - 273.15,
        }
    except Exception as e:
        print(f"  [skip] {date_str}: {e}")
        return None


def main():
    dates = pd.date_range(cfg.START_DATE, cfg.END_DATE, freq="D")
    rows = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        print(f"Fetching {date_str} ...")
        row = fetch_one_day(date_str)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.GFS_OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {cfg.GFS_OUT_CSV}")


if __name__ == "__main__":
    main()
