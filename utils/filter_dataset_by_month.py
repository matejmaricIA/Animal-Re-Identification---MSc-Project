import os
import shutil
import pandas as pd
import json
from datetime import datetime

# Paths
SOURCE_DIR = "../data/CameraTrapDataset-Processed/processed_data"
TARGET_DIR = "../data/camera_trap_dataset_filtered"
os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "animal_images"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "animal_crops"), exist_ok=True)

# Load timestamp data
df = pd.read_csv(os.path.join(SOURCE_DIR, "trail_cam_data.csv"))
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# Filter for April
april_df = df[df['datetime'].dt.month.isin([2, 3, 4, 5])]
april_images = set(april_df['filepath'].str.replace('\\', '/').str.split('/').str[-1])
april_image_stems = set(fname.split('.')[0] for fname in april_images)


# Copy matching animal_images
src_img = os.path.join(SOURCE_DIR, "animal_images")
dst_img = os.path.join(TARGET_DIR, "animal_images")
for fname in os.listdir(src_img):
    if fname in april_images:
        shutil.copy(os.path.join(src_img, fname), os.path.join(dst_img, fname))

# Copy matching animal_crops (match by stem)
src_crop = os.path.join(SOURCE_DIR, "animal_crops")
dst_crop = os.path.join(TARGET_DIR, "animal_crops")
for fname in os.listdir(src_crop):
    for stem in april_image_stems:
        if fname.startswith(stem):
            shutil.copy(os.path.join(src_crop, fname), os.path.join(dst_crop, fname))
            break


# Filter megadetector_results.json
with open(os.path.join(SOURCE_DIR, "megadetector_results.json"), "r") as f:
    md_data = json.load(f)

filtered_md = {
    "images": [img for img in md_data["images"] if os.path.basename(img["file"]) in april_images]
}
with open(os.path.join(TARGET_DIR, "megadetector_results.json"), "w") as f:
    json.dump(filtered_md, f, indent=2)

# Filter animal_detections.json
with open(os.path.join(SOURCE_DIR, "animal_detections.json"), "r") as f:
    det_data = json.load(f)

filtered_det = {k: v for k, v in det_data.items() if k in april_images}
with open(os.path.join(TARGET_DIR, "animal_detections.json"), "w") as f:
    json.dump(filtered_det, f, indent=2)

# Filter trail_cam_data.csv
april_df.to_csv(os.path.join(TARGET_DIR, "trail_cam_data.csv"), index=False)

print("April dataset created in:", TARGET_DIR)
