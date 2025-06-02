import json, os
from PIL import Image

IMAGE_ROOT       = "../data/camera_trap_dataset_filtered/animal_images"
YOLO_LABELS_DIR  = "../data/camera_trap_dataset_filtered/yolo_labels"
JSON_PATH        = "../data/camera_trap_dataset_filtered/speciesnet_perbox.json"
OUT_JSON_PATH    = "../data/camera_trap_dataset_filtered/speciesnet_perbox_updated.json"

CLASSES = [
    "wild boar",
    "roe deer",
    "red fox",
    "hare",
    "badger",
    "red squirrel",
    "blank"
]

with open(JSON_PATH) as f:
    data = json.load(f)

# filename → original JSON entry
entry_lookup = {os.path.basename(e["filepath"]): e for e in data["predictions"]}

n_img, n_box = 0, 0
for txt_file in os.listdir(YOLO_LABELS_DIR):
    if not txt_file.endswith(".txt"):
        continue

    stem      = os.path.splitext(txt_file)[0]
    img_file  = f"{stem}.JPG"                    # adjust if .jpg/.png mix
    img_path  = os.path.join(IMAGE_ROOT, img_file)
    label_path = os.path.join(YOLO_LABELS_DIR, txt_file)

    if img_file not in entry_lookup or not os.path.exists(img_path):
        continue

    # read image size once – needed to re-normalise if you ever want pixel boxes
    with Image.open(img_path) as im:
        W, H = im.size

    detections = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts)
            cls_id = int(cls_id)
            x = cx - w / 2
            y = cy - h / 2
            detections.append({
                "category": "1",
                "conf": 1.0,
                "bbox": [x, y, w, h],            # still normalised
                "classifications": {
                    "classes": [CLASSES[cls_id]],
                    "scores":  [1.0]
                }
            })
            n_box += 1

    # overwrite ONLY detections – timestamp & every other field stay untouched
    entry_lookup[img_file]["detections"] = detections
    n_img += 1

with open(OUT_JSON_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"Updated {n_img} images with {n_box} boxes.")
print("Timestamps and all other metadata are preserved.")
print("Saved:", OUT_JSON_PATH)
