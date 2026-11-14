"""Prepare calibration images for Hailo INT8 optimization."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


CALIBRATION_DIR = Path("calibration")
OUTPUT_DIR = Path("calibration_output")
NPY_FILE = OUTPUT_DIR / "starry_night_calibration_100.npy"
CSV_FILE = OUTPUT_DIR / "starry_night_calibration_100.csv"

IMAGE_SIZE = 256
MAX_CALIBRATION_IMAGES = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_files = []

for extension in ("*.jpg", "*.jpeg", "*.png"):
    image_files.extend(CALIBRATION_DIR.glob(extension))

image_files = sorted(set(image_files))[:MAX_CALIBRATION_IMAGES]

calibration_arrays = []
metadata_rows = []

for index, image_file in enumerate(image_files, start=1):
    with Image.open(image_file) as image:
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_array = np.asarray(image, dtype=np.float32)

    # The ONNX model expects RGB input values in the [-1, 1] range.
    image_array = image_array / 127.5 - 1.0

    calibration_arrays.append(image_array)
    metadata_rows.append(
        {
            "filename": image_file.name,
            "height": image_array.shape[0],
            "width": image_array.shape[1],
            "channels": image_array.shape[2],
            "min_value": float(image_array.min()),
            "max_value": float(image_array.max()),
            "mean_value": float(image_array.mean()),
        }
    )

    print(f"{index:03d} / {len(image_files):03d}: {image_file.name}")

calibration_dataset = np.stack(calibration_arrays).astype(np.float32)

expected_shape = (
    MAX_CALIBRATION_IMAGES,
    IMAGE_SIZE,
    IMAGE_SIZE,
    3,
)

if calibration_dataset.shape != expected_shape:
    raise ValueError(
        f"Expected calibration shape {expected_shape}, "
        f"but got {calibration_dataset.shape}"
    )

np.save(NPY_FILE, calibration_dataset)
pd.DataFrame(metadata_rows).to_csv(CSV_FILE, index=False)

print()
print("Calibration dataset complete")
print(f"Shape: {calibration_dataset.shape}")
print(f"dtype: {calibration_dataset.dtype}")
print(f"Range: {calibration_dataset.min():.4f} to {calibration_dataset.max():.4f}")
print(f"NPY: {NPY_FILE}")
print(f"CSV: {CSV_FILE}")
