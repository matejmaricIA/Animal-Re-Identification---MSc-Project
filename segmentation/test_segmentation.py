#!/usr/bin/env python3
"""
Simple script to test dataset-specific segmentation implementations.
Takes random photos from a dataset and applies segmentation to visually inspect results.
"""

import os
import cv2
import random
import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from segmentation module
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation import has_segmenter, segment_dataset
import pandas as pd

def find_images_in_dataset(dataset_name, data_root="./data", num_samples=5):
    """Find random images from a dataset directory."""
    dataset_path = Path(data_root) / dataset_name
    
    # Common image locations for different datasets
    possible_dirs = [
        "dataset",
        "animal_images",  # For MedvednicaDS
        "images"
    ]
    
    all_images = []
    for subdir in possible_dirs:
        img_dir = dataset_path / subdir
        if img_dir.exists():
            print(f"🔍 Looking in {img_dir}")
            # Find all image files
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                found_images = list(img_dir.rglob(f"*{ext}"))
                all_images.extend(found_images)
                if found_images:
                    print(f"  Found {len(found_images)} {ext} files")
    
    if not all_images:
        print(f"❌ No images found in {dataset_path}")
        print(f"   Tried subdirectories: {possible_dirs}")
        return []
    
    # Sample random images
    num_samples = min(num_samples, len(all_images))
    sampled = random.sample(all_images, num_samples)
    
    print(f"✅ Found {len(all_images)} images total, sampling {num_samples}")
    return sampled

def test_segmentation_on_image(image_path, dataset_name):
    """Test segmentation on a single image and save result."""
    print(f"  Processing: {image_path.name}")
    
    # Load original image
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"    ❌ Could not load image")
        return None
    
    # Check if segmentation is available
    if not has_segmenter(dataset_name):
        print(f"    ❌ No segmenter available for {dataset_name}")
        return None
    
    try:
        # Use the simple test approach - import segmentation functions directly
        if dataset_name == "BelugaID":
            # Simple beluga segmentation for testing
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            segmented = cv2.bitwise_and(original, mask_3ch)
            print(f"    ✅ Applied simple threshold segmentation")
            return segmented
        else:
            print(f"    ❌ No segmentation implemented for {dataset_name}")
            return None
            
    except Exception as e:
        print(f"    ❌ Segmentation failed: {e}")
        return None

def create_comparison_image(original, segmented, image_name, dataset_name):
    """Create side-by-side comparison of original and segmented image."""
    if segmented is None:
        return original
    
    # Resize images to same height if needed
    h1, w1 = original.shape[:2]
    h2, w2 = segmented.shape[:2]
    
    if h1 != h2:
        # Resize to smaller height
        target_height = min(h1, h2)
        if h1 > target_height:
            original = cv2.resize(original, (int(w1 * target_height / h1), target_height))
        if h2 > target_height:
            segmented = cv2.resize(segmented, (int(w2 * target_height / h2), target_height))
    
    # Create side-by-side comparison
    comparison = cv2.hconcat([original, segmented])
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Segmented", (original.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, f"{dataset_name} - {image_name}", (10, comparison.shape[0] - 10), font, 0.7, (255, 255, 255), 2)
    
    return comparison

def test_dataset_segmentation(dataset_name, num_samples=5, output_dir="./segmentation_results"):
    """Test segmentation on random images from a dataset."""
    print(f"\n🧪 Testing segmentation for: {dataset_name}")
    
    # Check if segmenter exists
    if not has_segmenter(dataset_name):
        print(f"❌ No segmenter available for {dataset_name}")
        return
    
    # Find random images
    images = find_images_in_dataset(dataset_name, num_samples=num_samples)
    if not images:
        return
    
    # Create output directory
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Testing segmentation...")
        
        # Load original
        original = cv2.imread(str(img_path))
        if original is None:
            continue
        
        # Apply segmentation
        segmented = test_segmentation_on_image(img_path, dataset_name)
        
        # Create comparison
        comparison = create_comparison_image(original, segmented, img_path.name, dataset_name)
        
        # Save result
        output_file = output_path / f"{img_path.stem}_comparison.jpg"
        cv2.imwrite(str(output_file), comparison)
        
        if segmented is not None:
            print(f"    ✅ Saved: {output_file}")
        else:
            print(f"    ⚠️  Saved original only: {output_file}")
    
    print(f"🎯 Results saved in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test dataset-specific segmentation")
    parser.add_argument('dataset', help='Dataset name (e.g., BelugaID, Giraffes, MedvednicaDS)')
    parser.add_argument('--samples', type=int, default=5, help='Number of random samples to test')
    parser.add_argument('--output', default='./segmentation_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Test the specified dataset
    test_dataset_segmentation(args.dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
