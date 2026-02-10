#!/usr/bin/env python3
import random
import argparse
import sys
from pathlib import Path

import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation import has_segmenter, segment_image
from preprocessing import (
    SOFT_MASK_SIGMA,
    SOFT_MASK_BG,
    SOFT_MASK_ERODE_PX,
    SOFT_MASK_BG_BLUR_SIGMA,
)


def find_random_images(dataset_name, data_root="./data", num_samples=5):
    dataset_path = Path(data_root) / dataset_name
    possible_dirs = ["dataset", "animal_images", "images"]

    all_images = []
    for subdir in possible_dirs:
        img_dir = dataset_path / subdir
        if not img_dir.exists():
            continue
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
            all_images.extend(list(img_dir.rglob(f"*{ext}")))

    if not all_images:
        print(f"No images found in {dataset_path}")
        return []

    num_samples = min(num_samples, len(all_images))
    sampled = random.sample(all_images, num_samples)
    print(f"Found {len(all_images)} images, sampling {num_samples}")
    return sampled


def test_segmentation(dataset_name, num_samples=5, output_dir="./segmentation_test_results"):
    print(f"\nTesting segmentation for: {dataset_name}")
    print(
        f"Soft mask constants: sigma={SOFT_MASK_SIGMA}, bg={SOFT_MASK_BG}, "
        f"erode_px={SOFT_MASK_ERODE_PX}, bg_blur_sigma={SOFT_MASK_BG_BLUR_SIGMA}"
    )

    if not has_segmenter(dataset_name):
        print(f" No segmenter available for {dataset_name}")
        return

    images = find_random_images(dataset_name, num_samples=num_samples)
    if not images:
        return

    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing: {img_path.name}")
        original = cv2.imread(str(img_path))
        if original is None:
            print("     Could not load image")
            continue

        hard = segment_image(dataset_name, original.copy())
        if hard is None:
            print("     Segmentation failed (hard mask)")
            continue

        soft = segment_image(
            dataset_name,
            original.copy(),
            soft_mask_sigma=SOFT_MASK_SIGMA,
            soft_mask_bg=SOFT_MASK_BG,
            soft_mask_erode_px=SOFT_MASK_ERODE_PX,
            soft_mask_bg_blur_sigma=SOFT_MASK_BG_BLUR_SIGMA,
        )
        if soft is None:
            print("     Segmentation failed (soft mask)")
            continue

        h, w = original.shape[:2]
        if w > 1000:
            scale = 1000 / w
            new_w, new_h = int(w * scale), int(h * scale)
            original = cv2.resize(original, (new_w, new_h))
            hard = cv2.resize(hard, (new_w, new_h))
            if soft is not None:
                soft = cv2.resize(soft, (new_w, new_h))

        panels = [original, hard]
        labels = ["Original", "HardMask"]
        panels.append(soft)
        labels.append("SoftMask")
        comparison = cv2.hconcat(panels)
        font = cv2.FONT_HERSHEY_SIMPLEX
        x = 10
        for j, label in enumerate(labels):
            cv2.putText(comparison, label, (x, 30), font, 1, (0, 255, 0), 2)
            x += panels[j].shape[1]
        cv2.putText(
            comparison,
            f"{dataset_name} - {img_path.name}",
            (10, comparison.shape[0] - 10),
            font,
            0.6,
            (255, 255, 255),
            2,
        )

        output_file = output_path / f"{img_path.stem}_comparison.jpg"
        cv2.imwrite(str(output_file), comparison)
        print(f"     Saved: {output_file}")

    print(f"\nResults saved in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Grounded SAM2 segmentation testing")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--output", default="./segmentation_test_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    test_segmentation(args.dataset, args.samples, args.output)


if __name__ == "__main__":
    main()
