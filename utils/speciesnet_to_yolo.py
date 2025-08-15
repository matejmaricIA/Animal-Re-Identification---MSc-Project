#!/usr/bin/env python3
"""
Convert SpeciesNet classification output to YOLO format for LabelImg annotation verification.

Usage:
    python speciesnet_to_yolo.py --input animals_classified.json --output_dir yolo_labels --images_dir animal_images

The script:
1. Reads SpeciesNet JSON output (already cleaned and classified)
2. Extracts the last word from classification strings (e.g., "red fox" from full taxonomy)
3. Creates YOLO format annotation files (.txt) for each image
4. Creates a classes.txt file with all unique species found
5. Optionally copies images to output directory for easy LabelImg use
"""

import json
import os
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image

def extract_species_name(classification_string: str) -> str:
    """
    Extract the species name (last word) from SpeciesNet classification string.
    
    Example: "ac0e8ba7-7261-4d17-8645-11ed3d02165a;mammalia;carnivora;canidae;vulpes;vulpes;red fox"
    Returns: "red fox"
    """
    if not classification_string:
        return "unknown"
    
    # Split by semicolon and get the last part
    parts = classification_string.split(';')
    if parts:
        species_name = parts[-1].strip()
        return species_name if species_name else "unknown"
    return "unknown"

def convert_bbox_to_yolo(bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from [x_center, y_center, width, height] format to YOLO format.
    
    Args:
        bbox: [x_center, y_center, width, height] in normalized coordinates [0,1]
    
    Returns:
        (x_center, y_center, width, height) in normalized coordinates [0,1] (same as input)
    """
    # The bounding box is already in the correct format for YOLO!
    # bbox format: [x_center_norm, y_center_norm, width_norm, height_norm]
    return tuple(bbox)

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return 1920, 1080  # Default dimensions if image can't be read

def process_speciesnet_output(input_json: str, output_dir: str, images_dir: str, copy_images: bool = True) -> None:
    """
    Process SpeciesNet JSON output and create YOLO format annotations.
    
    Args:
        input_json: Path to SpeciesNet animals_classified.json file
        output_dir: Directory to save YOLO format files
        images_dir: Directory containing the original images
        copy_images: Whether to copy images to output directory
    """
    
    # Create output directories
    labels_dir = Path(output_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    if copy_images:
        images_output_dir = Path(output_dir) / "images"
        images_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SpeciesNet results
    print(f"Loading SpeciesNet results from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Track all unique species found
    all_species: Set[str] = set()
    
    # Process each image
    processed_count = 0
    images_with_detections = 0
    
    # First pass: collect all species
    for prediction in data.get('predictions', []):
        detections = prediction.get('detections', [])
        
        for detection in detections:
            classifications = detection.get('classifications', {})
            classes = classifications.get('classes', [])
            if classes:
                # Get the best classification (first one with highest score)
                best_classification = classes[0]
                species_name = extract_species_name(best_classification)
                all_species.add(species_name)
    
    # Create species to class ID mapping
    species_list = sorted(list(all_species))
    species_to_id = {species: idx for idx, species in enumerate(species_list)}
    
    # Save classes.txt file
    classes_file = Path(output_dir) / "classes.txt"
    with open(classes_file, 'w') as f:
        for species in species_list:
            f.write(f"{species}\n")
    
    print(f"Found {len(species_list)} unique species:")
    for i, species in enumerate(species_list):
        print(f"  {i}: {species}")
    
    # Second pass: create YOLO annotations
    print(f"\nCreating YOLO annotations...")
    for prediction in data.get('predictions', []):
        filepath = prediction.get('filepath', '')
        detections = prediction.get('detections', [])
        
        if not detections:
            continue
            
        # Get image name without extension for label file
        image_name = Path(filepath).stem
        label_file = labels_dir / f"{image_name}.txt"
        
        # Get full image path
        full_image_path = Path(images_dir) / filepath
        
        if not full_image_path.exists():
            print(f"Warning: Image not found: {full_image_path}")
            continue
        
        # Note: No need to get image dimensions since bbox coordinates are already normalized
        
        # Process detections
        yolo_lines = []
        
        for detection in detections:
            classifications = detection.get('classifications', {})
            classes = classifications.get('classes', [])
            bbox = detection.get('bbox')
            
            if not bbox or len(bbox) != 4 or not classes:
                continue
            
            # Get the best classification (first one with highest score)
            best_classification = classes[0]
            species_name = extract_species_name(best_classification)
            class_id = species_to_id.get(species_name, 0)
            
            x_center, y_center, width, height = convert_bbox_to_yolo(bbox)
            
            # YOLO format: class_id x_center y_center width height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save annotation file
        if yolo_lines:
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            images_with_detections += 1
            
            # Copy image if requested
            if copy_images:
                dest_image = images_output_dir / Path(filepath).name
                shutil.copy2(full_image_path, dest_image)
        
        processed_count += 1
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")
    
    # Create summary
    summary_file = Path(output_dir) / "conversion_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"SpeciesNet to YOLO Conversion Summary\n")
        f.write(f"=====================================\n\n")
        f.write(f"Input file: {input_json}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Images directory: {images_dir}\n\n")
        f.write(f"Total images processed: {processed_count}\n")
        f.write(f"Images with detections: {images_with_detections}\n")
        f.write(f"Unique species found: {len(species_list)}\n\n")
        f.write(f"Species list:\n")
        for i, species in enumerate(species_list):
            f.write(f"  {i}: {species}\n")
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 YOLO labels saved to: {labels_dir}")
    print(f"📄 Classes file: {classes_file}")
    if copy_images:
        print(f"🖼️  Images copied to: {images_output_dir}")
    print(f"📊 Summary: {summary_file}")
    print(f"\nTotal: {images_with_detections} images with detections from {processed_count} processed images")

def main():
    parser = argparse.ArgumentParser(description="Convert SpeciesNet output to YOLO format")
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to SpeciesNet animals_classified.json file')
    parser.add_argument('--output_dir', '-o', required=True,
                       help='Output directory for YOLO format files')
    parser.add_argument('--images_dir', '-d', required=True,
                       help='Directory containing the original images')
    parser.add_argument('--no-copy-images', action='store_true',
                       help='Do not copy images to output directory')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
        
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1
    
    copy_images = not args.no_copy_images
    
    try:
        process_speciesnet_output(
            args.input, 
            args.output_dir, 
            args.images_dir,
            copy_images
        )
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

if __name__ == "__main__":
    exit(main())