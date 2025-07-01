if SEGMENTATION_MODEL_TYPE in ('isnet', 'combined'):
    session = new_session(ISNET_MODEL_NAME)
if SEGMENTATION_MODEL_TYPE in ('sam', 'combined'):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import torch
    _DEVICE = 'cuda' if DEVICE.upper() == 'GPU' and torch.cuda.is_available() else 'cpu'
    _sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(_DEVICE)
    mask_generator = SamAutomaticMaskGenerator(_sam)
elif SEGMENTATION_MODEL_TYPE == 'sam2':
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator   # ← NEW

    _DEVICE = "cuda" if (DEVICE.upper() == "GPU" and torch.cuda.is_available()) else "cpu"

    _sam2_core = build_sam2(
        SAM2_CFG,
        SAM2_CHECKPOINT_PATH,
        device=_DEVICE,
        dtype=torch.bfloat16,      # or .half() / .float()
    )

    # one automatic-mask generator you’ll reuse for every image
    amg = SAM2AutomaticMaskGenerator(_sam2_core, points_per_side=64) 
    #print(sam2_predictor)
else:
    session = None

def _binary_mask(img):          # expects uint8 [H,W] where FG=255
    return (img > 0).astype(np.uint8)

def _iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or (mask_a, mask_b).sum()
    return 0.0 if union == 0 else inter / union

def _combined_isnet_sam(image):
    """Return FG-masked image using Eq.(1) from the paper."""
    # 1) ISNet mask (single full-image mask, good recall)
    _, buf = cv2.imencode('.png', image)
    isnet_rgba = remove(image, session=session)
    if not isinstance(isnet_rgba, np.ndarray):
        isnet_rgba = cv2.cvtColor(np.array(isnet_rgba), cv2.COLOR_RGBA2BGRA)
    mask_isnet = _binary_mask(isnet_rgba[:, :, 3])           # alpha → binary

    # 2) SAM masks (many accurate proposals)
    sam_proposals = mask_generator.generate(image)
    if not sam_proposals:        # SAM failed → fall back to ISNet
        return cv2.bitwise_and(image, image, mask=mask_isnet * 255)

    # 3) keep proposals that overlap enough with ISNet
    kept = []
    for m in sam_proposals:
        mask = _binary_mask(m["segmentation"])
        if _iou(mask, mask_isnet) >= SAM_ISNET_IOU_THETA:
            kept.append(mask)

    # 4) union the kept masks, or fall back if nothing passed Θ
    if kept:
        merged = np.zeros_like(mask_isnet, dtype=np.uint8)
        for m in kept:
            merged = np.logical_or(merged, m)
        final_mask = merged.astype(np.uint8)
    else:
        final_mask = mask_isnet

    return cv2.bitwise_and(image, image, mask=final_mask * 255)

def mantiuk_tone_mapping(image):
    
    #image = cv2.imread(os.path.join('./data/ATRW', image_path))
    
    tonemapMantiuk = cv2.createTonemapMantiuk(scale = 0.7, saturation = 0.7)
    image = image.astype(np.float32) / 255.0
    
    mantiuk_image = tonemapMantiuk.process(image)
    mantiuk_image = np.clip(mantiuk_image * 255, 0,  255).astype(np.uint8)
    
    return mantiuk_image

def background_removal(image):
    """Apply background removal using the configured segmentation model."""
    if SEGMENTATION_MODEL_TYPE == 'isnet':
        _, buffer = cv2.imencode('.png', image)
        background_removed = remove(buffer.tobytes(), session=session)
        processed_image = cv2.imdecode(np.frombuffer(background_removed, np.uint8), cv2.IMREAD_UNCHANGED)
        return processed_image
    elif SEGMENTATION_MODEL_TYPE == 'sam':
        masks = mask_generator.generate(image)
        if not masks:
            return image
        largest = max(masks, key=lambda m: m['area'])
        mask = largest['segmentation'].astype(np.uint8) * 255
        return cv2.bitwise_and(image, image, mask=mask)
    elif SEGMENTATION_MODEL_TYPE == 'sam2':
        # -- SAM-2 AMG expects RGB numpy --
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get a list of dicts; each has 'segmentation', 'area', ...
        masks = amg.generate(rgb)

        if not masks:                       # no mask found
            return image

        largest = max(masks, key=lambda m: m['area'])
        mask    = largest['segmentation'].astype(np.uint8) * 255

        return cv2.bitwise_and(image, image, mask=mask)

    elif SEGMENTATION_MODEL_TYPE == 'combined':
        return _combined_isnet_sam(image)

    else:
        print(f"Unsupported segmentation model type: {SEGMENTATION_MODEL_TYPE}")
        return image
def process_image(row, output_dir, use_mantiuk, dataset_name, remove_background):
    """ Process an image by applying Mantiuk tone mapping and background removal."""

    #image_path = os.path.join(f'./data/{dataset_name}', row['path'])
    image_path = os.path.join(f'{WILD_DATASET_PATH}', row['path'])
    identity = row['identity']
    #split = row['original_split']
    id = str(row['image_id'])
    save_dir = os.path.join(output_dir, str(identity))
    os.makedirs(save_dir, exist_ok = True)

    image = cv2.imread(image_path)
    if use_mantiuk:
        image = mantiuk_tone_mapping(image)
    
    if remove_background:
        masked_image = background_removal(image)
    else:
        masked_image = image
    cv2.imwrite(os.path.join(save_dir, f'{id}.jpg'), masked_image)

    return save_dir


def preprocess_dataset(df, output_dir, dataset_name, use_mantiuk = True, remove_background = True):
    """
    Preprocess the dataset by applying Mantiuk tone mapping
    and background removal using SAM and ISNet."""
    
    args = [(row, output_dir, use_mantiuk, dataset_name, remove_background) for _, row in df.iterrows()]
    
    # Process sequentially
    processed_paths = []
    #index = 0
    for arg in tqdm(args, desc = "Preprocessing images", unit = "image"):
        processed_path = process_image(*arg)
        processed_paths.append(processed_path)
        #print(f"Processed image {index}/{len(args)}")
        #index += 1
    if remove_background:
        df['processed_path_segmented'] = processed_paths
    else:
        df['processed_path'] = processed_paths
    return df
        
def preprocess_inference(image_paths, use_mantiuk=False, remove_background=False):
    save_dir = os.path.join(TMP, 'segmented')
    os.makedirs(save_dir, exist_ok = True)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if use_mantiuk:
            image = mantiuk_tone_mapping(image)
        
        if remove_background:
            masked_image = background_removal(image)
        else:
            masked_image = image

        cv2.imwrite(os.path.join(save_dir, f'{os.path.basename(image_path)}'), masked_image)

    return save_dir

        
if __name__ == '__main__':
    print('...')


from multiprocessing import Pool, cpu_count
import os
import cv2
import numpy as np
from rembg import remove, new_session
import argparse
from constants import *
from tqdm import tqdm
import cv2, torch, numpy as np, json, os
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from rembg import remove, new_session
import pandas as pd
import torch

# hyper-params
MAX_SIDE              = 1440              # long side after resize
POINTS_PER_SIDE       = 20                # AMG density
MIN_REL_AREA          = 0.1           # skip blobs <5 % of frame
USE_HALF              = True              # half/ bfloat16 on GPU

_DEVICE = 'cuda' if DEVICE.upper() == 'GPU' and torch.cuda.is_available() else 'cpu'

# SAM core
_sam_core = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH) \
               .to(_DEVICE).eval()



    
_predictor = SamPredictor(_sam_core)
_amg = SamAutomaticMaskGenerator(
    _sam_core,
    points_per_side=20,       
    pred_iou_thresh=0.75,        
    stability_score_thresh=0.85, 
    min_mask_region_area=int(0.05 * MAX_SIDE * MAX_SIDE),
    output_mode="binary_mask")


# ISNet session for fallback or stand-alone use
session = new_session(ISNET_MODEL_NAME) \
          if SEGMENTATION_MODEL_TYPE in ('isnet', 'combined') else None

def _pre_enhance(img_bgr):
    # 1) resize ↓
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w*s), int(h*s)), cv2.INTER_AREA)

    # 2) contrast on V channel (CLAHE is slower; equalHist is OK here)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3) mild noise suppression
    img_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=60, sigmaSpace=25)

    # 4) light sharpening
    img_bgr = cv2.addWeighted(img_bgr, 1.0, 
                              cv2.GaussianBlur(img_bgr, (5,5), 3), -0.6, 0)

    return img_bgr

# Helper functions
def _resize(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image

def _select_mask(props, rgb):
    """Score each mask; higher score = more likely the animal."""
    h, w = rgb.shape[:2]
    cx, cy = w / 2, h / 2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    edge = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200)

    best, best_score = None, -1
    for p in props:
        m = p["segmentation"]
        area = m.sum()
        if area == 0:
            continue

        # ❶ centrality — prefer masks whose centroid is near image centre
        ys, xs = np.nonzero(m)
        dist = np.hypot(xs.mean() - cx, ys.mean() - cy) / max(h, w)  # 0…1

        # ❷ texture / colour — animals have higher sat & edges than sky/grass
        tex = sat[m].mean() / 255.0 + edge[m].mean() / 255.0

        # ❸ SAM’s own quality estimate
        iou  = p["predicted_iou"]

        score = 1.2 * (1 - dist) + 1.0 * tex + 1.5 * iou
        if score > best_score:
            best, best_score = m, score
    return best

@torch.inference_mode()
def _segment(image_bgr: np.ndarray, box_xyxy=None) -> np.ndarray:
    """
    Return binary mask {0,1}. Prefers SAM; falls back to ISNet if SAM fails.
    `box_xyxy` – list/tuple [x1,y1,x2,y2] in original image coords, or None.
    """
    image_bgr = _resize(image_bgr)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    #img_s = _pre_enhance(image_bgr)
    #rgb   = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    mask = None
    if SEGMENTATION_MODEL_TYPE.startswith('sam'):
        try:
            if box_xyxy is not None:                              # fast, accurate
                _predictor.set_image(rgb)
                masks, *_ = _predictor.predict(
                    box=np.asarray(box_xyxy, dtype=np.int32),
                    multimask_output=False)
                mask = masks[0]
            else:                                                 # automatic masks
                props = _amg.generate(rgb)
                if props:
                    mask = _select_mask(props, rgb)
                    #k = 2                          # merge up to two best masks
                    #top = sorted(props, key=lambda p: p["predicted_iou"], reverse=True)[:k]
                    #mask = np.logical_or.reduce([p["segmentation"] for p in top]).astype(np.uint8)

        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA out of memory error caught: {e}")
                print("Clearing GPU cache and falling back to ISNet...")
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Set mask to None to trigger ISNet fallback
                mask = None
            else:
                # Re-raise if it's a different RuntimeError
                raise e

    if mask is None and session is not None:                  # SAM failed or CUDA OOM
        alpha = remove(rgb, session=session)[:, :, 3] > 0
        mask = alpha

    return mask.astype(np.uint8)                              # {0,1}

def _apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # 1) ensure H×W match the image
    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(mask,                               # nearest so 0/1 stay 0/1
                          (image_bgr.shape[1], image_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    # 2) ensure uint8 {0,255}
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = mask * 255
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def mantiuk_tone_mapping(image_bgr: np.ndarray) -> np.ndarray:
    tonemap = cv2.createTonemapMantiuk(scale=0.7, saturation=0.7)
    img_f32 = image_bgr.astype(np.float32) / 255.0
    out = tonemap.process(img_f32)
    return np.clip(out * 255, 0, 255).astype(np.uint8)

# --------------------------- main API ---------------------------- #
def process_image(row, out_dir, use_mantiuk=False, remove_bg=True):
    """
    Row-wise processing function. Expects at least:
        row['path']      – relative path to image file
        row['identity']  – class / individual id
        row['image_id']  – unique numeric id
    Optional:
        row['bbox']      – [x1,y1,x2,y2] (dataset already cropped? leave None)
    """
    img_path = Path(WILD_DATASET_PATH) / row['path']
    img      = cv2.imread(str(img_path))

    if use_mantiuk:
        img = mantiuk_tone_mapping(img)

    if remove_bg:
        box = row.get('bbox', None)               # may be missing
        mask = _segment(img, box)
        img  = _apply_mask(img, mask)

    save_dir = Path(out_dir) / str(row['identity'])
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / f"{row['image_id']}.jpg"), img)
    return str(save_dir)

def preprocess_dataset(df: pd.DataFrame,
                       output_dir: str,
                       use_mantiuk=False,
                       remove_background=True):
    """
    Applies tone mapping and/or foreground segmentation to every row in `df`.
    Adds a new column with the processed path(s).
    """
    processed_paths = []
    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="Preprocessing",
                       unit="img"):
        processed_paths.append(
            process_image(row, output_dir,
                          use_mantiuk=use_mantiuk,
                          remove_bg=remove_background))

    colname = 'processed_path_segmented' if remove_background else 'processed_path'
    df[colname] = processed_paths
    return df

def preprocess_inference(image_paths, use_mantiuk=False, remove_background=False):
    """
    Mask / tone-map a list of image files (no DataFrame needed).
    Saves into TMP/segmented and returns that directory path.
    """
    out_dir = Path(TMP) / "segmented"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if use_mantiuk:
            img = mantiuk_tone_mapping(img)
        if remove_background:
            mask = _segment(img)
            img  = _apply_mask(img, mask)
        cv2.imwrite(str(out_dir / Path(img_path).name), img)
    return str(out_dir)

if __name__ == '__main__':
    main()

