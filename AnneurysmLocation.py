import os
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import ast

def load_localizer_annotations(localizer_csv, series_id):
    df = pd.read_csv(localizer_csv)
    df = df[df['SeriesInstanceUID'] == series_id]
    df['coordinates'] = df['coordinates'].apply(ast.literal_eval)
    return df

def show_annotated_slices(series_dir, series_id, localizer_csv):
    annotations = load_localizer_annotations(localizer_csv, series_id)
    if annotations.empty:
        print(f"No annotations found for series {series_id}")
        return

    # Build a map of SOPInstanceUID → file path
    sop_to_file = {}
    for f in os.listdir(series_dir):
        if not f.endswith(".dcm"):
            continue
        dcm = pydicom.dcmread(os.path.join(series_dir, f), stop_before_pixels=True)
        sop_to_file[dcm.SOPInstanceUID] = f

    for _, row in annotations.iterrows():
        sop_uid = row['SOPInstanceUID']
        coords = row['coordinates']
        label = row['location']

        if sop_uid not in sop_to_file:
            print(f"SOPInstanceUID {sop_uid} not found in {series_dir}")
            continue

        dcm_path = os.path.join(series_dir, sop_to_file[sop_uid])
        dcm = pydicom.dcmread(dcm_path)
        image = dcm.pixel_array

        plt.imshow(image, cmap='gray')
        plt.scatter(coords['x'], coords['y'], c='red', s=60, label=label)
        plt.title(f"Annotated Aneurysm ({label})")
        plt.axis('off')
        plt.legend()
        plt.show()

# === USAGE ===
series_id = "1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317"
series_dir = "series"
show_annotated_slices(series_dir, series_id, "train_localizers.csv")
