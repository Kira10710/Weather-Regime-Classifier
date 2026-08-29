"""
Central config for the project. Edit these values, everything else reads from here.
"""

# --- Region: bounding box (lat/lon) ---
# Default: Maharashtra (roughly). Change if you picked a different region on Day 0.
REGION_NAME = "Maharashtra"
LAT_MIN, LAT_MAX = 15.6, 22.0
LON_MIN, LON_MAX = 72.6, 80.9

# --- Date range for historical data pull ---
# Default: one monsoon season (keeps data volume small + regime signal strong)
START_DATE = "2023-06-01"
END_DATE = "2023-09-30"

# --- Forecast lead time ---
# 72 = Day-3 forecast (GFS forecast hours are in multiples of 3, e.g. 0,3,6...384)
LEAD_HOURS = 72

# --- GFS run cycle to use each day (00z is standard) ---
RUN_HOUR = "00"

# --- Variables ---
# APCP = total precipitation (accumulated), TMP = temperature
GFS_VARS = ["APCP", "TMP"]

# --- Output paths ---
GFS_OUT_CSV = "gfs_forecast.csv"
IMERG_OUT_CSV = "imerg_observed.csv"
JOINED_OUT_CSV = "joined_dataset.csv"
REGIME_LABELS_CSV = "regime_labels.csv"  # you fill this in manually (see build_dataset.py)
