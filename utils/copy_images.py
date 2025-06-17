import os
import shutil

SOURCE_DIR = "../data/ATRW/segmented_dataset"
TARGET_DIR = "../data/ds_test"

os.makedirs(TARGET_DIR, exist_ok=True)

for subfolder in os.listdir(SOURCE_DIR):
    full_subdir = os.path.join(SOURCE_DIR, subfolder)
    if not os.path.isdir(full_subdir):
        continue

    for fname in os.listdir(full_subdir):
        src_path = os.path.join(full_subdir, fname)
        dst_path = os.path.join(TARGET_DIR, f"{subfolder}_{fname}")
        shutil.copy2(src_path, dst_path)
