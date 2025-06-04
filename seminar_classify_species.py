"""
1.  Read MegaDetector JSON (detections.json).
2.  Crop every bbox -> temp folder of JPG crops.
3.  Run SpeciesNet CLI once on that crop folder with --country HRV.
4.  Stitch the per-crop predictions back into the original JSON
    + inject timestamps from the CSV.
5.  Write speciesnet_perbox.json (ready for analysis).

Uses:
    python classify_cli_perbox.py \
        --images_dir      data/.../animal_images \
        --detections_json data/.../detections.json \
        --metadata_csv    data/.../trail_cam_data.csv \
        --out_json        data/.../speciesnet_perbox.json \
        --country HRV \
        --batch_size 16
"""

import argparse, json, shutil, subprocess, tempfile, uuid
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

# crops
def save_crop(img_path, bbox, dest_dir, stem, idx):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x, y, w, h = map(float, bbox)
    crop = img.crop((x*W, y*H, (x+w)*W, (y+h)*H))
    crop_name = f"{stem}_crop{idx}.jpg"
    crop.save(dest_dir / crop_name, "JPEG", quality=90)
    return crop_name  # filename only

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",      required=True, type=Path)
    ap.add_argument("--detections_json", required=True, type=Path)
    ap.add_argument("--metadata_csv",    required=True, type=Path)
    ap.add_argument("--out_json",        required=True, type=Path)
    ap.add_argument("--country",         default="HRV")
    ap.add_argument("--batch_size",      type=int, default=16)
    args = ap.parse_args()

    # load detections
    dets_json = json.load(args.detections_json.open())
    preds_root = dets_json.get("predictions") or dets_json["images"]

    # tmp dir for crops
    crops_dir = Path(tempfile.mkdtemp(prefix="crops_"))
    mapping = {}  # crop_name -> (frame_rec, det_idx)

    print("=== Cropping bounding boxes ===")
    for rec in tqdm(preds_root, desc="frames"):
        stem = Path(rec["filepath"]).name.rsplit(".", 1)[0]
        img_path = args.images_dir / Path(rec["filepath"]).name
        for i, det in enumerate(rec["detections"]):
            crop_name = save_crop(img_path, det["bbox"], crops_dir, stem, i)
            mapping[crop_name] = (rec, i)

    # run SpeciesNet CLI once on the crops folder
    crop_pred_json = Path(tempfile.gettempdir()) / f"crops_pred_{uuid.uuid4().hex}.json"
    cmd = [
        "python", "-m", "speciesnet.scripts.run_model",
        "--folders", str(crops_dir),
        "--predictions_json", str(crop_pred_json),
        "--batch_size", str(args.batch_size),
        "--model", "kaggle:google/speciesnet/pyTorch/v4.0.1a",
        "--country", args.country,
    ]
    print("\n=== Running SpeciesNet CLI on crops ===")
    subprocess.run(cmd, check=True)
    print("=== CLI done ===\n")

    # load crop predictions
    crop_preds = json.load(crop_pred_json.open())["predictions"]
    for cp in crop_preds:
        crop_file = Path(cp["filepath"]).name
        classes   = cp["classifications"]["classes"]
        scores    = cp["classifications"]["scores"]

        frame_rec, det_idx = mapping[crop_file]
        frame_rec["detections"][det_idx]["classifications"] = {
            "classes": classes,
            "scores" : scores
        }

    # inject timestamps
    meta = pd.read_csv(args.metadata_csv)
    meta["basename"] = meta["filepath"].apply(lambda p: Path(p).name)
    ts_map = dict(zip(meta["basename"], meta["datetime"].astype(str)))
    for rec in preds_root:
        base = Path(rec["filepath"]).name
        if base in ts_map:
            rec["timestamp"] = ts_map[base]

    # 6) save final JSON
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump({"predictions": preds_root}, f, indent=2)

    # 7) clean up crops
    shutil.rmtree(crops_dir, ignore_errors=True)

    print(f"Wrote per-box species and timestamps to{args.out_json}")

if __name__ == "__main__":
    main()
