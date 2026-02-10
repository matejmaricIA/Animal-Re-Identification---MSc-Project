from multiprocessing import Pool, cpu_count
import os
import cv2
import numpy as np
import argparse
from constants import *
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from segmentation import segment_image, has_segmenter, segment_dataset


# Soft mask controls (feathered edge) to avoid harsh black boundaries that attract keypoints.
# Set SOFT_MASK_SIGMA=0.0 to disable (hard mask).
SOFT_MASK_SIGMA = 2.0
SOFT_MASK_BG = "black"          # "mean" | "black" | "blur"
SOFT_MASK_ERODE_PX = 2         # helps reduce halos at the boundary
SOFT_MASK_BG_BLUR_SIGMA = 25.0 # only used when SOFT_MASK_BG="blur"

# Mantiuk tone-mapper is expensive to construct; reuse a singleton.
_TONEMAP_MANTIUK = cv2.createTonemapMantiuk(scale=0.8, saturation=0.8, gamma=1.0)

def mantiuk_tone_mapping(image):
    
    #image = cv2.imread(os.path.join('./data/ATRW', image_path))
    
    image = image.astype(np.float32) / 255.0
    
    mantiuk_image = _TONEMAP_MANTIUK.process(image)
    mantiuk_image = np.clip(mantiuk_image * 255, 0,  255).astype(np.uint8)
    
    return mantiuk_image

def background_removal(image, dataset_name=None):
    """Apply background removal using Grounded SAM2 for supported datasets."""
    if not dataset_name:
        raise ValueError("dataset_name is required for segmentation.")
    segmented = segment_image(
        dataset_name,
        image,
        soft_mask_sigma=SOFT_MASK_SIGMA,
        soft_mask_bg=SOFT_MASK_BG,
        soft_mask_erode_px=SOFT_MASK_ERODE_PX,
        soft_mask_bg_blur_sigma=SOFT_MASK_BG_BLUR_SIGMA,
    )
    if segmented is None:
        raise ValueError(
            f"No segmentation prompt configured for dataset '{dataset_name}'."
        )
    return segmented
def process_image(row, output_dir, use_mantiuk, dataset_name, remove_background):
    """ Process an image by applying Mantiuk tone mapping and background removal."""

    # Prefer local paths when they exist; otherwise fall back to WildlifeReID10k root.
    project_root = Path(__file__).resolve().parent
    local_candidate = project_root / str(row["path"])
    if local_candidate.exists():
        image_path = str(local_candidate)
    else:
        image_path = str(project_root / WILD_DATASET_PATH / str(row["path"]))
    
    identity = row['identity']
    id = str(row['image_id'])
    save_dir = os.path.join(output_dir, str(identity))
    os.makedirs(save_dir, exist_ok = True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    if use_mantiuk:
        image = mantiuk_tone_mapping(image)
    
    if remove_background:
        masked_image = background_removal(image, dataset_name)
    else:
        masked_image = image
    cv2.imwrite(os.path.join(save_dir, f'{id}.jpg'), masked_image)

    return save_dir


def preprocess_dataset(
    df,
    output_dir,
    dataset_name,
    use_mantiuk=True,
    remove_background=True,
):
    """
    Preprocess the dataset by applying Mantiuk tone mapping
    and background removal using Grounded SAM2."""
    # Path to metadata CSV for the given dataset
    metadata_path = DATAFRAME_PATH.format(dataset_name)

    # Load existing metadata if available so we don't lose previously
    # processed path columns when updating one of them.
    if os.path.exists(metadata_path):
        #existing_df = pd.read_csv(metadata_path).set_index("image_id")
        existing_df = pd.read_csv(
            metadata_path, dtype={"image_id": str}
        ).set_index("image_id")
    else:
        #existing_df = pd.DataFrame().set_index("image_id")
        existing_df = pd.DataFrame(columns=["image_id"]).set_index("image_id")
        
    df["image_id"] = df["image_id"].astype(str)
    if remove_background and not has_segmenter(dataset_name):
        raise ValueError(
            f"Segmentation requested but no prompt configured for dataset '{dataset_name}'."
        )
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


def prepare_processed_dataset(
    dataset_name: str,
    df_raw: pd.DataFrame,
    *,
    remove_background: bool = False,
    use_mantiuk: bool = False,
    require_processed_paths: bool = False,
    log_prefix: str = "",
) -> tuple[pd.DataFrame, str, str]:
    """Load cached processed metadata or build it via preprocessing/segmentation."""
    prefix = f"{log_prefix} " if log_prefix else ""
    sub_dir = f"./data/{dataset_name}"
    os.makedirs(sub_dir, exist_ok=True)

    if df_raw is None:
        raise ValueError(f"{prefix}Missing dataset metadata (df_raw).")

    df_raw = df_raw.copy()
    df_raw["image_id"] = df_raw["image_id"].astype(str)
    csv_path = f"{sub_dir}/processed_metadata.csv"
    output_dir = f"{sub_dir}/segmented_dataset" if remove_background else f"{sub_dir}/dataset"

    df = None
    missing_cols = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"image_id": str})
        if require_processed_paths:
            required_cols = ["processed_path_segmented"] if remove_background else ["processed_path"]
            missing_cols = [
                col
                for col in required_cols
                if col not in df.columns or df[col].isna().all()
            ]

    if require_processed_paths:
        needs_preprocess = bool(missing_cols) or not os.path.exists(csv_path) or not os.path.exists(output_dir)
    else:
        needs_preprocess = not (os.path.exists(csv_path) and os.path.exists(output_dir))

    if needs_preprocess:
        if remove_background and has_segmenter(dataset_name) and not os.path.exists(output_dir):
            if prefix:
                print(f"{prefix}Segmenting dataset...")
            df = segment_dataset(
                df_raw.copy(),
                f"{output_dir}/",
                dataset_name,
                use_mantiuk=use_mantiuk,
            )
        else:
            df = preprocess_dataset(
                df_raw.copy(),
                f"{output_dir}/",
                dataset_name,
                use_mantiuk=use_mantiuk,
                remove_background=remove_background,
            )
        df.to_csv(csv_path, index=False)
    elif df is None:
        df = pd.read_csv(csv_path, dtype={"image_id": str})

    df["image_id"] = df["image_id"].astype(str)
    return df, csv_path, output_dir
        
def preprocess_inference(
    image_paths,
    use_mantiuk=False,
    remove_background=False,
    dataset_name=None,
):
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
