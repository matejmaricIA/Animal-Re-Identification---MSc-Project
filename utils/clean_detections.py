import json
from pathlib import Path

input_path = Path("../data/MedvednicaDS/detections.json")
output_path = Path("../data/MedvednicaDS/detections_cleaned.json")

with input_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Build reverse map from category id -> name (e.g., "1"->"animal")
cat_map = data.get("detection_categories", {})
# Normalize keys to strings
cat_map = {str(k): v for k, v in cat_map.items()}

def is_animal(det):
    # Category can be an id like "1" or possibly the string name "animal"
    cat = det.get("category")
    if cat is None:
        return False
    # If category is an id, map it; else compare directly
    name = cat_map.get(str(cat), None)
    if name is not None:
        return name == "animal"
    # Fallback if detector already stored the label
    return str(cat).lower() == "animal"

for pred in data.get("predictions", []):
    dets = pred.get("detections", [])
    # Condition: any detection with conf<0.5 OR not animal
    if any((d.get("conf", 0) <= 0.5) or (not is_animal(d)) for d in dets):
        pred["detections"] = []

with output_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Wrote cleaned JSON to {output_path}")
