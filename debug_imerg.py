"""
Debug helper - run this to see what's actually inside the downloaded
IMERG file, instead of guessing.

    python debug_imerg.py
"""

import earthaccess
import os
import config as cfg

earthaccess.login()

results = earthaccess.search_data(
    short_name="GPM_3IMERGDF",
    temporal=("2023-07-15", "2023-07-15"),
    bounding_box=(cfg.LON_MIN, cfg.LAT_MIN, cfg.LON_MAX, cfg.LAT_MAX),
)
print(f"Found {len(results)} granule(s)")

files = earthaccess.download(results, local_path="./imerg_raw")
print(f"Downloaded to: {files}")

path = files[0]
size_bytes = os.path.getsize(path)
print(f"File size: {size_bytes} bytes ({size_bytes / 1024:.1f} KB)")

if size_bytes < 10_000:
    print("\n[!] File is suspiciously small - likely an error page, not real data.")
    print("First 500 characters of file content:")
    with open(path, "rb") as f:
        print(f.read(500))
else:
    print("\nFile size looks reasonable. Inspecting internal structure...")
    try:
        import netCDF4
        ds = netCDF4.Dataset(path)
        print(f"Top-level groups: {list(ds.groups.keys())}")
        print(f"Top-level variables: {list(ds.variables.keys())}")
        for group_name in ds.groups:
            print(f"  Variables in '{group_name}': {list(ds.groups[group_name].variables.keys())}")
    except Exception as e:
        print(f"netCDF4 inspection failed: {e}")
