import os
import numpy as np
import cv2
import argparse
import sys
import pandas as pd
import torch
from pathlib import Path
import tqdm
from tqdm import trange
import tempfile
import shutil
# Create HDF5 file for storing descriptors
import h5py
import numpy as np


# For keynet, hardnet, and affnet
from kornia.feature import KeyNetAffNetHardNet
import kornia as K

from constants import (
    MODEL_PATH,
    DATAFRAME_PATH,
    SAVE_TEST_DESCRIPTORS_PATH,
    MAX_KEYPOINTS,
)

try:
    import lightglue
    from lightglue import SuperPoint, ALIKED, DoGHardNet, SIFT
    from lightglue.utils import load_image
    _LIGHTGLUE_AVAILABLE = True

except Exception:
    _LIGHTGLUE_AVAILABLE = False



def get_segmentation_tag(remove_background: bool) -> str:
    """Return a string tag for segmented or unsegmented mode."""
    return "segmented" if remove_background else "unsegmented"

def _parse_image_item(item):
    """Return (image_id, path) from either a path string or (id, path) pair."""
    if isinstance(item, (tuple, list)) and len(item) == 2:
        img_id, path = item
        return str(img_id), str(path)
    path = str(item)
    return Path(path).stem, path


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, max_size=None):
        self.paths = paths
        self.max_size = max_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_id, path = _parse_image_item(self.paths[index])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)

        h, w = img.shape[:2]
        scale_x = 1.0
        scale_y = 1.0

        max_size = self.max_size
        if max_size is not None and max_size > 0:
            max_dim = max(h, w)
            if max_dim > max_size:
                scale = max_size / float(max_dim)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                scale_x = new_w / float(w)
                scale_y = new_h / float(h)

        tens = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return img_id, tens, (scale_x, scale_y)


def get_image_paths(df, remove_background = True):
    if remove_background:
        img_locations = (df['processed_path_segmented'] + "/" + df['image_id'].apply(str) + '.jpg').tolist()
        return img_locations
    img_locations = (df['processed_path'] + "/" + df['image_id'].apply(str) + '.jpg').tolist()
    #print(img_locations)
    return img_locations


def extract_features(image_paths, model_path, output_dir, max_keypoints=MAX_KEYPOINTS):
    """Extract DISK features."""

    if _LIGHTGLUE_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = lightglue.DISK(max_num_keypoints=max_keypoints).to(device).eval()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        desc_h5_path = Path(output_dir) / "descriptors.h5"
        kp_h5_path = Path(output_dir) / "keypoints.h5"

        with h5py.File(desc_h5_path, "w") as desc_h5, h5py.File(kp_h5_path, "w") as kp_h5:
            for img_item in tqdm.tqdm(image_paths, desc="DISK features"):
                img_id, img_path = _parse_image_item(img_item)
                image = load_image(img_path).to(device)

                with torch.inference_mode():
                    try:
                        feats = extractor.extract(image)
                    except Exception as e:
                        print(f"Error extracting features for {img_path}: {e}")
                        continue

                desc = feats["descriptors"]
                kp = feats["keypoints"]
                if len(desc.shape) == 3:
                    desc = desc.squeeze(0)
                    kp = kp.squeeze(0)

                desc_np = desc.cpu().numpy().astype(np.float32)
                kp_np = kp.cpu().numpy().astype(np.float32)

                desc_h5.create_dataset(img_id, data=desc_np, compression="gzip")
                kp_h5.create_dataset(img_id, data=kp_np, compression="gzip")

                torch.cuda.empty_cache()

        print(f"DISK features saved to {desc_h5_path}")
        print(f"DISK keypoints saved to {kp_h5_path}")
        return

    # Old, deprecated way to use DISK, safe to delete.
    """sys.path.append('./disk/')
    from disk import DISK
    import detect
    
    dataset = detect.SceneDataset(image_paths, crop_size = (640, 640))
    state_dict = torch.load(model_path, map_location = 'cpu')
    weights = state_dict['extractor']
    model = DISK(window = 8, desc_dim=128)  
    model.load_state_dict(weights)
    model = model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    detect.extract(dataset, output_dir, model)
    #try:
    #    detect.extract(dataset, output_dir, model)
    #except Exception as e:
    #    print(f"Error: {e}")
    #    sys.exit(1)
    #finally:
    print("DISK feature extraction complete.")"""

def extract_features_lightglue(
    image_paths,
    output_dir,
    feature_type = "aliked",
    max_keypoints = MAX_KEYPOINTS,
):
    """Extract features using the LightGlue extractors.
    """

    if not _LIGHTGLUE_AVAILABLE:
        raise RuntimeError("LightGlue is not installed")

    extractors = {
        "superpoint": SuperPoint,
        #"disk": DISK,
        "aliked": ALIKED,
        "doghardnet": DoGHardNet,
        "sift": SIFT,
    }
    feature_type = feature_type.lower()
    if feature_type not in extractors:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Extractor = extractors[feature_type]
    extractor = Extractor(max_num_keypoints=max_keypoints).to(device).eval()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    desc_h5_path = Path(output_dir) / "descriptors.h5"
    kp_h5_path = Path(output_dir) / "keypoints.h5"

    with h5py.File(desc_h5_path, "w") as desc_h5, h5py.File(kp_h5_path, "w") as kp_h5:
        for img_item in tqdm.tqdm(image_paths, desc=f"{feature_type} features"):
            img_id, img_path = _parse_image_item(img_item)
            image = load_image(img_path).to(device)
            with torch.inference_mode():
                feats = extractor.extract(image)

            desc = feats["descriptors"]
            #desc = np.squeeze(array, axis = 0)
            #print(desc.shape)
            #if desc.shape[0] in (128, 256) and desc.shape[1] > desc.shape[0]:
            #    desc = desc.t().contiguous()             
            
            #desc_np = desc.cpu().numpy().astype(np.float32)
            desc_np = desc.squeeze(0).cpu().numpy().astype(np.float32)
            #desc_np = feats["descriptors"].T.cpu().numpy().astype(np.float32)
            kp_np   = feats["keypoints"].cpu().numpy().astype(np.float32)

            desc_h5.create_dataset(img_id, data=desc_np, compression="gzip")
            kp_h5.create_dataset(img_id, data=kp_np, compression="gzip")

            torch.cuda.empty_cache()

    print(f"{feature_type} features saved to {desc_h5_path}")
    print(f"{feature_type} keypoints saved to {kp_h5_path}")

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Feature extraction using DISK")
    parser.add_argument('--model', type = str, help = "Path to the model's .pth save file", default = MODEL_PATH)
    parser.add_argument('--df', type = str, help = "Path to the saved dataframe file", default = DATAFRAME_PATH)
    parser.add_argument('--output_dir', type = str, default = SAVE_TEST_DESCRIPTORS_PATH)

    args = parser.parse_args()
    
    img_paths = get_image_paths(args.df)
    extract_features(img_paths, args.model, args.output_dir)
