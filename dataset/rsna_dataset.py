import os
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Optional
import cv2

class RSNADataset(Dataset):
    def __init__(self, 
                 series_dir: str, 
                 labels_csv: str,
                 series_uids: list,
                 input_size=(64, 128, 128),
                 transform: Optional[callable] = None):
        """
        Args:
            series_dir (str): Path to the series/ directory.
            labels_csv (str): Path to the train.csv file.
            series_uids (list): List of SeriesInstanceUIDs to load.
            input_size (tuple): Desired 3D shape (D, H, W).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.series_dir = series_dir
        self.labels_df = pd.read_csv(labels_csv).set_index("SeriesInstanceUID")
        self.series_uids = series_uids
        self.input_size = input_size
        self.transform = transform

        # Extract 14 target columns
        self.label_columns = [
            col for col in self.labels_df.columns 
            if col not in ['PatientAge', 'PatientSex', 'Modality']
        ]
        
    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, idx):
        while True:
            series_uid = self.series_uids[idx]
            try:
                volume = self.load_dicom_volume(series_uid)
                volume = self.resize_volume(volume, self.input_size)
                volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-5)
                volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1, 1)  # [3, D, H, W]

                label = torch.tensor(self.labels_df.loc[series_uid, self.label_columns].values.astype(np.float32))

                if self.transform:
                    volume = self.transform(volume)

                return volume, label
            
            except ValueError as e:
                print(f"[WARN] Skipping {series_uid} due to error: {e}")
                idx = (idx + 1) % len(self.series_uids)


    def load_dicom_volume(self, series_uid):
        series_path = os.path.join(self.series_dir, series_uid)
        files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')]

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                pixel_array = dcm.pixel_array
                slices.append((int(dcm.InstanceNumber), pixel_array))
            except Exception as e:
                print(f"[WARN] Could not read pixel array for {f}: {e}")
                continue

        if len(slices) == 0:
            raise ValueError(f"No valid DICOM slices found for {series_uid}")

        # Sort slices by InstanceNumber
        slices.sort(key=lambda x: x[0])

        # Extract just the pixel data
        images = [s[1] for s in slices]

        # Optional: center crop number of slices to max_slices
        max_slices = 64
        if len(images) > max_slices:
            mid = len(images) // 2
            images = images[mid - max_slices // 2 : mid + max_slices // 2]

        # Filter out invalid slices before checking shapes or resizing
        clean_images = []
        for img in images:
            if img is None or img.size == 0:
                print(f"[WARN] Skipping empty slice in {series_uid}")
                continue
            if img.shape[0] == 0 or img.shape[1] == 0:
                print(f"[WARN] Skipping slice with shape {img.shape} in {series_uid}")
                continue
            clean_images.append(img)

        if len(clean_images) == 0:
            raise ValueError(f"No usable slices left for {series_uid}")

        # Ensure all slices have the same shape
        shapes = set(img.shape for img in clean_images)
        if len(shapes) > 1:
            print(f"[WARN] Skipping {series_uid} due to inconsistent slice shapes after filtering: {shapes}")
            raise ValueError("Inconsistent slice shapes")

        # Resize slices
        target_h, target_w = 64, 64
        images_resized = []
        for img in clean_images:
            try:
                resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                images_resized.append(resized)
            except Exception as e:
                print(f"[WARN] Failed to resize slice in {series_uid}: {e}")
                continue

        if len(images_resized) == 0:
            raise ValueError(f"All slice resizing failed for {series_uid}")

        # Stack into [D, H, W] volume
        volume = np.stack(images_resized, axis=0).astype(np.float32)

        # Normalize intensity
        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-5)

        return volume




    def resize_volume(self, volume, target_shape):
        """Resize 3D volume to (D, H, W)."""
        if isinstance(volume, np.ndarray):
            volume = torch.tensor(volume, dtype=torch.float32)
        if volume.ndim == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        elif volume.ndim == 4:
            volume = volume.unsqueeze(0)  # already has channel
        else:
            raise ValueError(f"Unexpected volume shape: {volume.shape}")

        volume_resized = F.interpolate(volume, size=target_shape, mode='trilinear', align_corners=False)
        return volume_resized.squeeze().numpy()



# Example utility
def get_series_uids_with_labels(labels_csv: str):
    df = pd.read_csv(labels_csv)
    return df['SeriesInstanceUID'].tolist()

def load_labels(labels_csv: str):
    df = pd.read_csv(labels_csv)
    label_columns = [
        col for col in df.columns 
        if col not in ['PatientAge', 'PatientSex', 'Modality', 'SeriesInstanceUID']
    ]
    label_dict = {}
    for _, row in df.iterrows():
        uid = row["SeriesInstanceUID"]
        labels = row[label_columns].values.astype(np.float32)
        label_dict[uid] = labels
    return label_dict, label_columns

