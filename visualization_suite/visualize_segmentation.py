#!/usr/bin/env python3
"""
Generate a thesis-quality figure visualizing the Grounded SAM 2 segmentation pipeline.
Stages: Original -> GroundingDINO Box -> SAM 2 Mask -> Final Soft-Masked Output.
"""

import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from segmentation.grounded_sam2 import (
    _predict_boxes,
    _predict_best_mask,
    _clean_mask,
    _apply_soft_mask,
    DINO_BOX_THRESHOLD,
    DINO_TEXT_THRESHOLD
)
from visualization_suite import collage, io, style

def draw_box(image, box, label=""):
    """Draw a single box on the image."""
    vis = image.copy()
    h, w = vis.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    
    # Draw rectangle
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Draw label background
    if label:
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(vis, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return vis

def draw_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """Draw a semi-transparent mask overlay."""
    vis = image.copy()
    mask_bool = mask > 0
    vis[mask_bool] = vis[mask_bool] * (1 - alpha) + np.array(color) * alpha
    
    # Draw contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 2)
    return vis

def main():
    parser = argparse.ArgumentParser(description="Generate segmentation pipeline visualization.")
    parser.add_argument("--image", type=str, 
                        default=str(project_root / "data/cowdataset/dataset/CowDataset_1/34466.jpg"),
                        help="Path to the input image.")
    parser.add_argument("--prompt", type=str, default="cow . cattle",
                        help="Text prompt for GroundingDINO.")
    parser.add_argument("--out", type=str, 
                        default=str(project_root / "visualization_suite/output/segmentation_pipeline_vis.png"),
                        help="Output filename.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    print(f"Processing {args.image} with prompt '{args.prompt}'...")

    # 1. Load Image
    original_img = cv2.imread(args.image)
    if original_img is None:
        print("Failed to load image.")
        return

    # 2. GroundingDINO Detection
    print("Running GroundingDINO...")
    boxes = _predict_boxes(original_img, args.prompt, DINO_BOX_THRESHOLD, DINO_TEXT_THRESHOLD)
    
    if boxes is None or len(boxes) == 0:
        print("No boxes detected.")
        return

    # Select the most confident/central box (simplified logic: largest area)
    # _boxes_to_xyxy returns numpy array
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = np.argmax(areas)
    target_box = boxes[best_idx]
    
    # Visualization: Box
    box_vis = draw_box(original_img, target_box, label=args.prompt.split(".")[0])

    # 3. SAM 2 Segmentation
    print("Running SAM 2...")
    # _predict_best_mask takes all boxes but we want to visualize the process for the best one
    # But internally it iterates. Let's pass all boxes and let it pick the best mask as per implementation
    mask = _predict_best_mask(original_img, boxes)
    
    if mask is None:
        print("No mask generated.")
        return
    
    # Visualization: Mask Overlay
    mask_vis = draw_mask_overlay(original_img, mask)

    # 4. Post-processing (Clean + Soft Mask)
    print("Post-processing...")
    mask_u8 = _clean_mask(mask)
    final_vis = _apply_soft_mask(
        original_img, 
        mask_u8, 
        feather_sigma=2.0, 
        bg_mode="mean", 
        bg_blur_sigma=25.0, 
        erode_px=2
    )

    # 5. Assemble Collage using Visualization Suite
    print("Creating visualization...")
    
    images = [original_img, box_vis, mask_vis, final_vis]
    titles = [
        "(a) Original Input", 
        "(b) GroundingDINO Detection", 
        "(c) SAM 2 Segmentation", 
        "(d) Final Soft-Masked Output"
    ]

    # Use make_grid from visualization_suite
    # We want a 1x4 row or 2x2 grid. 1x4 is good for a wide figure, 2x2 for a column.
    # Let's go with 1x4 for the "process flow" feel.
    grid_img, _ = collage.make_grid(images, titles=titles, cols=4, figsize=(20, 5))

    # Save
    io.save_image(args.out, grid_img)
    print(f"Saved visualization to {args.out}")

if __name__ == "__main__":
    main()
