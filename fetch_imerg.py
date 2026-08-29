"""
Fetch IMERG (satellite-gauge merged rainfall) ground-truth data for your
region/date-range - this is the "actual weather" side of your dataset.

SETUP (run once, on your own machine):
    pip install earthaccess xarray pandas h5netcdf

    Then run this once in a Python shell to log in interactively
    (it'll prompt for your free NASA Earthdata username/password
    and cache a token locally):
        import earthaccess
        earthaccess.login()

WHAT THIS DOES:
    For each day in your date range, finds and downloads the daily
    IMERG Final Run product, subsets to your region's bounding box,
    and averages precipitation over that box.

NOTE:
    Like fetch_gfs.py, this needs real internet access - won't run
    in a sandboxed/offline environment.
"""

import earthaccess
import xarray as xr
import pandas as pd

import config as cfg


def fetch_one_day(date_str: str) -> dict | None:
    try:
        results = earthaccess.search_data(
            short_name="GPM_3IMERGDF",  # IMERG Final Run, daily
            temporal=(date_str, date_str),
            bounding_box=(cfg.LON_MIN, cfg.LAT_MIN, cfg.LON_MAX, cfg.LAT_MAX),
        )
        if not results:
            print(f"  [skip] {date_str}: no IMERG granule found")
            return None

        files = earthaccess.download(results, local_path="./imerg_raw")
        # V07B files have variables at the top level, no "Grid" group
        ds = xr.open_dataset(files[0])

        box = ds["precipitation"].sel(
            lat=slice(cfg.LAT_MIN, cfg.LAT_MAX),
            lon=slice(cfg.LON_MIN, cfg.LON_MAX),
        )
        precip_mm = float(box.mean().values)  # .mean() with no dims averages over all axes regardless of order

        return {
            "valid_date": date_str,
            "region": cfg.REGION_NAME,
            "observed_precip_mm": precip_mm,
        }
    except Exception as e:
        print(f"  [skip] {date_str}: {e}")
        return None


def main():
    earthaccess.login()  # uses cached token after first interactive login

    dates = pd.date_range(cfg.START_DATE, cfg.END_DATE, freq="D")
    rows = []
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        print(f"Fetching {date_str} ...")
        row = fetch_one_day(date_str)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.IMERG_OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {cfg.IMERG_OUT_CSV}")


if __name__ == "__main__":
    main()