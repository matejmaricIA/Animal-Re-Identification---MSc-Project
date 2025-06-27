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

from constants import (MODEL_PATH, DATAFRAME_PATH, DEVICE, SAVE_TEST_DESCRIPTORS_PATH, SAVE_TRAIN_DESCRIPTORS_PATH,
 PATCH_SIZE, MULTISCALE_SCALES, MAX_FEATURES_PER_SCALE, ENABLE_MULTISCALE)

try:
    import lightglue
    from lightglue import SuperPoint, ALIKED, DoGHardNet, SIFT
    from lightglue.utils import load_image
    _LIGHTGLUE_AVAILABLE = True

except Exception:
    _LIGHTGLUE_AVAILABLE = False

import torch.nn.functional as F

class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            path = self.paths[index]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(path)
            tens = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            return Path(path).stem, tens


def get_image_paths(df):
    img_locations = (df['processed_path'] + "/" + df['image_id'].apply(str) + '.jpg').tolist()
    
    #print(img_locations)
    return img_locations


def extract_features(image_paths, model_path, output_dir, max_keypoints=5000):
    """Extract DISK features with optional multi-scale support."""

    if _LIGHTGLUE_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = lightglue.DISK(max_num_keypoints=max_keypoints).to(device).eval()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        desc_h5_path = Path(output_dir) / "descriptors.h5"
        kp_h5_path = Path(output_dir) / "keypoints.h5"

        with h5py.File(desc_h5_path, "w") as desc_h5, h5py.File(kp_h5_path, "w") as kp_h5:
            for img_path in tqdm.tqdm(image_paths, desc="DISK features"):
                img_id = Path(img_path).stem
                image = load_image(img_path).to(device)

                desc_list = []
                kp_list = []
                scales = MULTISCALE_SCALES if ENABLE_MULTISCALE else [1.0]
                for scale in scales:
                    if scale != 1.0:
                        scaled = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False).squeeze(0)
                    else:
                        scaled = image

                    with torch.inference_mode():
                        feats = extractor.extract(scaled)
                        #print(feats.keys())

                        desc = feats["descriptors"]
                        kp = feats["keypoints"]
                        if len(desc.shape) == 3:
                            desc = desc.squeeze(0)
                            kp = kp.squeeze(0)

                        

                        desc_np = desc.cpu().numpy().astype(np.float32)
                        kp_np = kp.cpu().numpy().astype(np.float32)
                        if scale != 1.0:
                            kp_np /= scale

                        if desc_np.shape[0] > MAX_FEATURES_PER_SCALE:
                            desc_np = desc_np[:MAX_FEATURES_PER_SCALE]
                            kp_np = kp_np[:MAX_FEATURES_PER_SCALE]

                        desc_list.append(desc_np)
                        kp_list.append(kp_np)

                if not desc_list:
                    continue

                desc_np = np.concatenate(desc_list, axis=0)
                kp_np = np.concatenate(kp_list, axis=0)

                desc_h5.create_dataset(img_id, data=desc_np, compression="gzip")
                kp_h5.create_dataset(img_id, data=kp_np, compression="gzip")

                torch.cuda.empty_cache()

        print(f"DISK features saved to {desc_h5_path}")
        print(f"DISK keypoints saved to {kp_h5_path}")
        return


    sys.path.append('./disk/')
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
    print("DISK feature extraction complete.")
    
def extract_features_keynet_hardnet(image_paths,
                                    output_dir,
                                    num_features= 5000,
                                    batch_size = 1,
                                    use_half = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    lfeat  = KeyNetAffNetHardNet(num_features=num_features).to(device).eval()
    
    if use_half:
        lfeat = lfeat.half()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    desc_h5_path = Path(output_dir) / "descriptors.h5"
    kp_h5_path = Path(output_dir) / "keypoints.h5"

    with h5py.File(desc_h5_path, "w") as desc_h5, h5py.File(kp_h5_path, "w") as kp_h5:
        for img_idx in trange(len(image_paths), desc="KeyNetAffNetHardNet"):
            img_path = image_paths[img_idx]
            img_id   = Path(img_path).stem       # e.g. "00012345" without extension

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Cannot read {img_path}")
                continue

            tens = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
            tens = tens.to(device, dtype=torch.float16 if use_half else torch.float32)


            with torch.inference_mode():
                lafs, _, descs = lfeat(tens.to(device))

            
            desc_np = descs.squeeze(0).cpu().numpy().astype(np.float32)  # (N,128)
            if desc_np.shape[0] == 0:
                # zero keypoints
                continue
            
            # Extract keypoint coordinates from LAFs
            # LAFs shape: (1, N, 2, 3) - center coordinates are in [:, :, :, 2]
            keypoints_np = lafs[0, :, :, 2].cpu().numpy().astype(np.float32)  # (N, 2)

            # Save both descriptors and keypoints
            desc_h5.create_dataset(img_id, data=desc_np, compression="gzip")
            kp_h5.create_dataset(img_id, data=keypoints_np, compression="gzip")
            
            del tens, lafs, descs
            torch.cuda.empty_cache()

    print(f"KeyNet+AffNet+HardNet features saved to {desc_h5_path}")
    print(f"KeyNet+AffNet+HardNet keypoints saved to {kp_h5_path}")
    

def extract_features_keynet_hardnet_faster(
    image_paths,
    output_dir,
    num_features = 4500,
    batch_size = 4,
    num_workers = 8,
    use_half = True,
):
    """Extract KeyNet+AffNet+HardNet features.

    A small dataloader is used to load images in parallel which reduces
    overhead when a large number of files is processed.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lfeat = KeyNetAffNetHardNet(num_features=num_features).to(device).eval()

    if use_half:
        lfeat = lfeat.half()
        
    dataset = ImageDataset(image_paths)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x,  # return list of tuples
    )
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    desc_h5_path = Path(output_dir) / "descriptors.h5"
    kp_h5_path = Path(output_dir) / "keypoints.h5"
    
    dtype = torch.float16 if use_half else torch.float32
    
    with h5py.File(desc_h5_path, "w") as desc_h5, h5py.File(kp_h5_path, "w") as kp_h5:
        for batch in tqdm.tqdm(loader, desc="KeyNetAffNetHardNet"):
            for img_id, tens in batch:
                tens = tens.to(device, dtype=dtype)
                
                with torch.inference_mode():
                    lafs, _, descs = lfeat(tens.unsqueeze(0))
                    
                desc_np = descs.squeeze(0).cpu().numpy().astype(np.float32)
                if desc_np.shape[0] == 0:
                    continue
                keypoints_np = lafs[0, :, :, 2].cpu().numpy().astype(np.float32)
                
                desc_h5.create_dataset(img_id, data=desc_np, compression="gzip")
                kp_h5.create_dataset(img_id, data=keypoints_np, compression="gzip")
                
                del tens, lafs, descs
                torch.cuda.empty_cache()
                
    print(f"KeyNet+AffNet+HardNet features saved to {desc_h5_path}")
    print(f"KeyNet+AffNet+HardNet keypoints saved to {kp_h5_path}")

def extract_features_lightglue(
    image_paths,
    output_dir,
    feature_type = "aliked",
    max_keypoints = 4096,
):
    """Extract features using the LightGlue extractors.

    Parameters
    ----------
    image_paths : list[str]
        Paths to input images.
    output_dir : str or Path
        Directory where descriptor and keypoint files will be written.
    feature_type : str
        One of ``superpoint``, ``disk``, ``aliked``, ``doghardnet`` or ``sift``.
    max_keypoints : int
        Maximum number of keypoints per image.
    """

    if not _LIGHTGLUE_AVAILABLE:
        raise RuntimeError("LightGlue is not installed")

    extractors = {
        "superpoint": SuperPoint,
        "disk": DISK,
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
        for img_path in tqdm.tqdm(image_paths, desc=f"{feature_type} features"):
            img_id = Path(img_path).stem
            image = load_image(img_path).to(device)
            with torch.inference_mode():
                feats = extractor.extract(image)

            desc = feats["descriptors"]

            if desc.shape[0] in (128, 256) and desc.shape[1] > desc.shape[0]:
                desc = desc.t().contiguous()             
            
            desc_np = desc.cpu().numpy().astype(np.float32)

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


