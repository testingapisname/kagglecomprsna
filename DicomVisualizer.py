import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

def load_volume_from_files(dicom_dir):
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith(".dcm")]

    slices = []
    for f in files:
        dcm = pydicom.dcmread(f)
        slices.append(dcm)
    slices.sort(key=lambda x: float(x.InstanceNumber))  # sort slices properly

    volume = np.stack([s.pixel_array for s in slices], axis=0)
    volume = volume.astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    return volume

def show_middle_slice(volume):
    mid = volume.shape[0] // 2
    plt.imshow(volume[mid], cmap='gray')
    plt.title(f"Middle Slice (index {mid})")
    plt.axis('off')
    plt.show()

# === USAGE ===
dicom_dir = "series"  # This matches your folder
vol = load_volume_from_files(dicom_dir)
show_middle_slice(vol)
