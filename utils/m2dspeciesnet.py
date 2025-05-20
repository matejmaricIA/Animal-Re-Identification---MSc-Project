#!/usr/bin/env python3
"""
Turn a MegaDetector JSON into the *exact* format SpeciesNet
expects for --classifier_only, keeping EVERY bbox.

Assumes your images live in <root>/animal_images/*.JPG
and you will run SpeciesNet with:
    --folders <root>               (NOT the animal_images sub-dir!)
"""

from pathlib import Path
import argparse, json

def main():
    #ap = argparse.ArgumentParser()
    #ap.add_argument("--in_json",  required=True, type=Path,
    #                help="Original MegaDetector JSON")
    #ap.add_argument("--out_json", required=True, type=Path,
    #                help="SpeciesNet-ready JSON to write")
    #args = ap.parse_args()

    
    
    IN_JSON = Path('../data/camera_trap_dataset_filtered/megadetector_results.json')
    OUT_JSON = Path('../data/camera_trap_dataset_filtered/detections.json')

    md = json.load(IN_JSON.open())

    if "images" not in md:
        raise SystemExit("Input file lacks an 'images' key – wrong file?")

    md["predictions"] = [
        {
            "filepath"   : f"animal_images/{Path(im['file']).name}",   # match SpeciesNet
            "detections" : im["detections"]                           # ALL boxes kept
        }
        for im in md["images"]
    ]
    del md["images"]

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(md, f, indent=2)

    print(f"✅  Wrote {len(md['predictions'])} entries to {OUT_JSON}")

if __name__ == "__main__":
    main()
