"""
Run this FIRST, before fetch_gfs.py / fetch_imerg.py loop over the full
date range. Tests the pipeline end-to-end on a single date so you find
any setup/auth problems now, not halfway through Day 1.

    python test_pipeline.py
"""

import config as cfg
from fetch_gfs import fetch_one_day as fetch_gfs_one
from fetch_imerg import fetch_one_day as fetch_imerg_one
import earthaccess

TEST_DATE = "2023-07-15"  # any date well within your START/END range

print("=" * 50)
print("Testing GFS fetch...")
print("=" * 50)
gfs_row = fetch_gfs_one(TEST_DATE)
print(gfs_row)

print("\n" + "=" * 50)
print("Testing IMERG fetch...")
print("=" * 50)
earthaccess.login()
imerg_row = fetch_imerg_one(TEST_DATE)
print(imerg_row)

print("\n" + "=" * 50)
if gfs_row and imerg_row:
    print("SUCCESS - both sources work. Safe to run the full fetch scripts.")
else:
    print("FAILED - fix whichever source errored above before continuing.")
print("=" * 50)
