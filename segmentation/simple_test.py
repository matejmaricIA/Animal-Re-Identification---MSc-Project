#!/usr/bin/env python3
import os
import cv2
import random
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def beluga_segment(image):
    """Beluga segmentation using actual implementation."""
    try:
        from segmentation.beluga_segmentation import segment
        return segment(image)
    except Exception as e:
        print(f"Failed to import beluga segmentation: {e}")
        # Fallback to simple threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(image, image, mask=mask)

def giraffe_segment(image):
    """Simple giraffe segmentation - placeholder for your implementation."""
    # TODO: Implement actual giraffe segmentation
    return image  # Placeholder

def deer_segment(image):
    """Simple deer segmentation for MedvednicaDS - placeholder."""
    # TODO: Implement actual deer segmentation
    return image  # Placeholder

def wild_boar_segment(image):
    """Simple wild boar segmentation for MedvednicaDS - placeholder."""
    # TODO: Implement actual wild boar segmentation  
    return image  # Placeholder

def cow_segment(image):
    """Simple cow segmentation - placeholder."""
    # TODO: Implement actual cow segmentation
    return image  # Placeholder

def panda_segment(image):
    """Simple panda segmentation - placeholder."""
    # TODO: Implement actual panda segmentation
    return image  # Placeholder



def nyala_segment(image):
    """Nyala segmentation using actual implementation."""
    try:
        from segmentation.nyala_segmentation import segment
        return segment(image)
    except Exception as e:
        print(f"Failed to import nyala segmentation: {e}")
        # Fallback to simple processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(image, image, mask=mask)

def ipanda_segment(image):
    """IPanda50 segmentation using actual implementation."""
    try:
        from segmentation.ipanda_segmentation import segment
        return segment(image)
    except Exception as e:
        print(f"Failed to import ipanda segmentation: {e}")
        return image

def giraffe_segment(image):
    """Giraffe segmentation using actual implementation."""
    try:
        from segmentation.giraffe_segmentation import segment
        return segment(image)
    except Exception as e:
        print(f"Failed to import giraffe segmentation: {e}")
        return image

def hyena_segment(image):
    """Hyena segmentation using actual implementation."""
    try:
        from segmentation.hyena_segmentation import segment
        return segment(image)
    except Exception as e:
        print(f"Failed to import hyena segmentation: {e}")
        return image

def medvednica_segment(image):
    """MedvednicaDS segmentation using actual implementation."""
    try:
        from segmentation.medvednica_segmentation import segment
        return segment(image)
    except Exception as e:
        print(f"Failed to import medvednica segmentation: {e}")
        return image

# Mapping of dataset names to segmentation functions
SEGMENTERS = {
    'BelugaID': beluga_segment,
    'Giraffes': giraffe_segment,
    'roe_deer': medvednica_segment,
    'wild_boar': medvednica_segment,
    'CowDataset': cow_segment,
    'IPanda50': ipanda_segment,
    'HyenaID2022': hyena_segment,
    'NyalaData': nyala_segment,
}

def find_random_images(dataset_name, data_root="./data", num_samples=5):
    """Find random images from a dataset."""
    dataset_path = Path(data_root) / dataset_name
    
    # Common subdirectories where images might be stored
    possible_dirs = [
        "dataset",
        "animal_images",  # MedvednicaDS
        "images"
    ]
    
    all_images = []
    for subdir in possible_dirs:
        img_dir = dataset_path / subdir
        if img_dir.exists():
            # Find all image files recursively
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                all_images.extend(list(img_dir.rglob(f"*{ext}")))
    
    if not all_images:
        print(f"❌ No images found in {dataset_path}")
        return []
    
    # Sample random images
    num_samples = min(num_samples, len(all_images))
    sampled = random.sample(all_images, num_samples)
    
    print(f"✅ Found {len(all_images)} images in {dataset_name}, sampling {num_samples}")
    return sampled

def test_segmentation(dataset_name, num_samples=5, output_dir="./segmentation_test_results"):
    """Test segmentation on random images from a dataset."""
    print(f"\n🧪 Testing segmentation for: {dataset_name}")
    
    # Check if we have a segmenter for this dataset
    if dataset_name not in SEGMENTERS:
        print(f"❌ No segmenter available for {dataset_name}")
        print(f"Available datasets: {list(SEGMENTERS.keys())}")
        return
    
    segmenter = SEGMENTERS[dataset_name]
    
    # Find random images
    images = find_random_images(dataset_name, num_samples=num_samples)
    if not images:
        return
    
    # Create output directory
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing: {img_path.name}")
        
        # Load image
        original = cv2.imread(str(img_path))
        if original is None:
            print(f"    ❌ Could not load image")
            continue
        
        try:
            # Apply segmentation
            segmented = segmenter(original.copy())
            
            # Create side-by-side comparison
            # Resize if needed to fit side by side
            h, w = original.shape[:2]
            if w > 1000:  # Resize large images
                scale = 1000 / w
                new_w, new_h = int(w * scale), int(h * scale)
                original = cv2.resize(original, (new_w, new_h))
                segmented = cv2.resize(segmented, (new_w, new_h))
            
            # Create comparison
            comparison = cv2.hconcat([original, segmented])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Segmented", (original.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, f"{dataset_name} - {img_path.name}", 
                       (10, comparison.shape[0] - 10), font, 0.6, (255, 255, 255), 2)
            
            # Save result
            output_file = output_path / f"{img_path.stem}_comparison.jpg"
            cv2.imwrite(str(output_file), comparison)
            print(f"    ✅ Saved: {output_file}")
            
        except Exception as e:
            print(f"    ❌ Segmentation failed: {e}")
    
    print(f"\n🎯 All results saved in: {output_path}")
    print(f"📁 Open the folder to visually inspect the segmentation results!")

def main():
    parser = argparse.ArgumentParser(description="Simple segmentation testing")
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--output', default='./segmentation_test_results', help='Output directory')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        for dataset in SEGMENTERS.keys():
            print(f"  - {dataset}")
        return
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Test the dataset
    test_segmentation(args.dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
