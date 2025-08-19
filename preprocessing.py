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
from segmentation import segment_image

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

def background_removal(image, dataset_name=None):
    """Apply background removal using dataset-specific or configured model."""
    custom = segment_image(dataset_name, image) if dataset_name else None
    if custom is not None:
        return custom
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

    # Check if this is a manual dataset (wild_boar, roe_deer) or WildlifeReID10k dataset
    manual_datasets = ['wild_boar', 'roe_deer']
    
    if dataset_name.lower() in manual_datasets:
        # For manual datasets, use local project path
        image_path = os.path.join('.', row['path'])  # row['path'] already contains the full relative path
    else:
        # For WildlifeReID10k datasets, use the standard path
        image_path = os.path.join(f'{WILD_DATASET_PATH}', row['path'])
    
    identity = row['identity']
    id = str(row['image_id'])
    save_dir = os.path.join(output_dir, str(identity))
    os.makedirs(save_dir, exist_ok = True)

    image = cv2.imread(image_path)
    if use_mantiuk:
        image = mantiuk_tone_mapping(image)
    
    if remove_background:
        masked_image = background_removal(image, dataset_name)
    else:
        masked_image = image
    cv2.imwrite(os.path.join(save_dir, f'{id}.jpg'), masked_image)

    return save_dir


def preprocess_dataset(df, output_dir, dataset_name, use_mantiuk = True, remove_background = True):
    """
    Preprocess the dataset by applying Mantiuk tone mapping
    and background removal using SAM and ISNet."""
    # Path to metadata CSV for the given dataset
    metadata_path = DATAFRAME_PATH.format(dataset_name)

    # Load existing metadata if available so we don't lose previously
    # processed path columns when updating one of them.
    if os.path.exists(metadata_path):
        existing_df = pd.read_csv(metadata_path).set_index("image_id")
    else:
        #existing_df = pd.DataFrame().set_index("image_id")
        existing_df = pd.DataFrame(columns=["image_id"]).set_index("image_id")

    args = [(row, output_dir, use_mantiuk, dataset_name, remove_background) for _, row in df.iterrows()]

    # Process sequentially
    processed_paths = []
    for arg in tqdm(args, desc = "Preprocessing images", unit = "image"):
        processed_path = process_image(*arg)
        processed_paths.append(processed_path)

    if remove_background:
        df['processed_path_segmented'] = processed_paths
    else:
        df['processed_path'] = processed_paths

    # Merge with existing metadata to retain both processed path columns
    df = df.set_index("image_id")
    merged_df = df.combine_first(existing_df)

    # Ensure both processed path columns exist
    if 'processed_path' not in merged_df.columns:
        merged_df['processed_path'] = np.nan
    if 'processed_path_segmented' not in merged_df.columns:
        merged_df['processed_path_segmented'] = np.nan

    merged_df = merged_df.reset_index()

    # Save merged metadata so subsequent preprocessing runs keep both columns
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    merged_df.to_csv(metadata_path, index=False)

    return merged_df
        
def preprocess_inference(image_paths, use_mantiuk=False, remove_background=False, dataset_name=None):
    save_dir = os.path.join(TMP, 'segmented')
    os.makedirs(save_dir, exist_ok = True)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if use_mantiuk:
            image = mantiuk_tone_mapping(image)
        
        if remove_background:
            masked_image = background_removal(image, dataset_name)
        else:
            masked_image = image

        cv2.imwrite(os.path.join(save_dir, f'{os.path.basename(image_path)}'), masked_image)

    return save_dir

        
if __name__ == '__main__':
    print('...')
