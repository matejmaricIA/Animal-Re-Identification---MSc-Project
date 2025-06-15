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

# Create HDF5 file for storing descriptors
import h5py

# For keynet, hardnet, and affnet
from kornia.feature import KeyNetAffNetHardNet
import kornia as K

from constants import MODEL_PATH, DATAFRAME_PATH, DEVICE, SAVE_TEST_DESCRIPTORS_PATH, SAVE_TRAIN_DESCRIPTORS_PATH, PATCH_SIZE


def get_image_paths(df):
    img_locations = (df['processed_path'] + "/" + df['image_id'].apply(str) + '.jpg').tolist()
    
    #print(img_locations)
    return img_locations


def extract_features(image_paths, model_path, output_dir):
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
                                    num_features= 2500,
                                    batch_size = 1,
                                    use_half = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    lfeat  = KeyNetAffNetHardNet(num_features=num_features).to(device).eval()
    
    if use_half:
        lfeat = lfeat.half()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    h5_path = Path(output_dir) / "descriptors.h5"

    with h5py.File(h5_path, "w") as h5:
        for img_idx in trange(len(image_paths), desc="KeyNetAffNetHardNet"):
            img_path = image_paths[img_idx]
            img_id   = Path(img_path).stem       # e.g. "00012345" without extension

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] cannot read {img_path}")
                continue

            tens = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
            tens = tens.to(device, dtype=torch.float16 if use_half else torch.float32)


            with torch.inference_mode():
                lafs, _, descs = lfeat(tens.to(device))

            desc_np = descs.squeeze(0).cpu().numpy().astype(np.float32)  # (N,128)
            if desc_np.shape[0] == 0:
                # zero keypoints; mirror behaviour of DISK extractor (skip image)
                continue

            h5.create_dataset(img_id, data=desc_np, compression="gzip")
            
            del tens, lafs, descs
            torch.cuda.empty_cache()

    print(f"KeyNet+AffNet+HardNet features saved to {h5_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Feature extraction using DISK")
    parser.add_argument('--model', type = str, help = "Path to the model's .pth save file", default = MODEL_PATH)
    parser.add_argument('--df', type = str, help = "Path to the saved dataframe file", default = DATAFRAME_PATH)
    parser.add_argument('--output_dir', type = str, default = SAVE_TEST_DESCRIPTORS_PATH)

    args = parser.parse_args()
    
    img_paths = get_image_paths(args.df)
    extract_features(img_paths, args.model, args.output_dir)


