#!/usr/bin/env python3
"""
Create metadata for unlabeled animal datasets (wild_boar, roe_deer).

This creates basic metadata that matches exactly how the main pipeline works:
- Original images: data/{dataset}/original_data/
- Processed images: data/{dataset}/dataset/ (created by preprocessing)  
- Segmented images: data/{dataset}/segmented_dataset/ (created by preprocessing)

Usage:
    cd /home/mm-workstation/re-id/Animal-Re-Identification---MSc-Project
    python utils/metadata_generation.py wild_boar
    python utils/metadata_generation.py roe_deer
"""

import os
import sys
import hashlib
import pandas as pd
from pathlib import Path
import argparse

def generate_image_id(image_path, index):
    """Generate a unique image ID from the image path and index."""
    return hashlib.md5(f"{str(image_path)}_{index}".encode()).hexdigest()[:16]

def create_unlabeled_metadata(dataset_name):
    """Create metadata CSV for an unlabeled dataset."""
    
    # Project structure
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "data" / dataset_name
    original_data_dir = dataset_dir / "original_data"
    
    print(f"Looking for images in: {original_data_dir}")
    
    if not original_data_dir.exists():
        raise FileNotFoundError(f"Directory {original_data_dir} does not exist. Please create it and put your images there.")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


    image_files = sorted({
        p.resolve()
        for p in original_data_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in image_extensions
    })

    if not image_files:
        raise FileNotFoundError(f"No image files found in {original_data_dir}")

    print(f"Found {len(image_files)} unique images")
    
    # Create metadata entries
    metadata_rows = []
    
    for i, img_path in enumerate(sorted(image_files)):
        # Create relative path from project root (this is what the main script expects)
        rel_path = img_path.relative_to(project_root)
        
        metadata_rows.append({
            'image_id': generate_image_id(img_path, i),  # Make each ID unique
            'identity': f"unknown_{i:06d}",  # Dummy identity for unlabeled data
            'path': str(rel_path),  # Relative path from project root
            'dataset': dataset_name,  # Dataset name
            'split': 'all',  # Single split for counting
        })
    
    # Create DataFrame
    df = pd.DataFrame(metadata_rows)
    
    # Save metadata to the dataset directory (not inside original_data)
    metadata_path = dataset_dir / "processed_metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    print(
        "Metadata saved. Note: current `--count` mode requires GT `identity` labels "
        "to simulate the human oracle. For unlabeled datasets, a manual pair-vetting "
        "workflow is planned but not yet integrated."
    )
    
    return metadata_path

def main():
    parser = argparse.ArgumentParser(description="Create metadata for unlabeled datasets")
    parser.add_argument("dataset_name", help="Name of the dataset (e.g., wild_boar, roe_deer)")
    
    args = parser.parse_args()
    
    try:
        create_unlabeled_metadata(args.dataset_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
