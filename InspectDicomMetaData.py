import os
import pydicom

series_dir = "series"

for f in os.listdir(series_dir):
    if not f.endswith(".dcm"):
        continue

    path = os.path.join(series_dir, f)
    dcm = pydicom.dcmread(path, stop_before_pixels=True)

    print(f"Filename: {f}")
    print(f"  SeriesInstanceUID: {dcm.SeriesInstanceUID}")
    print(f"  SOPInstanceUID:    {dcm.SOPInstanceUID}")
    print("-" * 60)
