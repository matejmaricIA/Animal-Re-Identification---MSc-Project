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

from constants import MODEL_PATH, DATAFRAME_PATH, DEVICE, SAVE_TEST_DESCRIPTORS_PATH, SAVE_TRAIN_DESCRIPTORS_PATH, PATCH_SIZE, MULTISCALE_SCALES


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
    
def extract_features_multiscale_disk(image_paths, model_path, output_dir, scales=[0.5, 1.0, 1.5]):
    """
    Extract DISK features at multiple scales using different crop sizes.
    Much simpler than manual scaling!
    """
    sys.path.append('./disk/')
    from disk import DISK
    import detect
    import h5py
    import tempfile
    import shutil
    
    # Calculate crop sizes for each scale
    base_crop_size = 640  # Your working base size
    crop_sizes = []
    temp_dirs = []
    
    for scale in scales:
        crop_size = int(base_crop_size * scale)
        # Ensure it's a multiple of 16 for U-Net
        #crop_size = ((crop_size + 15) // 16) * 16
        crop_sizes.append((crop_size, crop_size))
        
        # Create temp directory for this scale
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        
        print(f"Extracting features at scale {scale} with crop_size {crop_size}x{crop_size}")
        
        # Use your existing function with different crop size
        dataset = detect.SceneDataset(image_paths, crop_size=(crop_size, crop_size))
        state_dict = torch.load(model_path, map_location='cpu')
        weights = state_dict['extractor']
        model = DISK(window=8, desc_dim=64) # Reducing desc_dim to 64 for memory efficiency
        model.load_state_dict(weights) 
        model = model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        
        detect.extract(dataset, temp_dir, model)
    
    # Combine features from all scales
    os.makedirs(output_dir, exist_ok=True)
    desc_h5_path = os.path.join(output_dir, "descriptors.h5")
    kp_h5_path = os.path.join(output_dir, "keypoints.h5")  
    
    with h5py.File(desc_h5_path, "w") as desc_h5, h5py.File(kp_h5_path, "w") as kp_h5:
        # Group features by image
        image_features = {}
        image_keypoints = {}
        
        for i, temp_dir in enumerate(temp_dirs):
            scale_desc_path = os.path.join(temp_dir, "descriptors.h5")
            scale_kp_path = os.path.join(temp_dir, "keypoints.h5") 
            
            if os.path.exists(scale_desc_path):
                with h5py.File(scale_desc_path, "r") as scale_h5:
                    for img_id in scale_h5.keys():
                        descriptors = scale_h5[img_id][:]
                        
                        if img_id not in image_features:
                            image_features[img_id] = []
                        image_features[img_id].append(descriptors)
                        
                
                if os.path.exists(scale_kp_path):
                    with h5py.File(scale_kp_path, "r") as kp_scale_h5:
                        for img_id in kp_scale_h5.keys():
                            keypoints = kp_scale_h5[img_id][:]
                            
                            if img_id not in image_keypoints:
                                image_keypoints[img_id] = []
                            image_keypoints[img_id].append(keypoints)
        
        # Save combined features
        for img_id, desc_list in image_features.items():
            if desc_list:
                combined_descriptors = np.vstack(desc_list)
                
                # Optional: Limit features to prevent memory issues
                max_features = 5000
                if len(combined_descriptors) > max_features:
                    indices = np.random.choice(len(combined_descriptors), max_features, replace=False)
                    combined_descriptors = combined_descriptors[indices]
                    
                    
                    if img_id in image_keypoints and image_keypoints[img_id]:
                        combined_keypoints = np.vstack(image_keypoints[img_id])
                        combined_keypoints = combined_keypoints[indices]
                    else:
                        combined_keypoints = None
                else:
                    # NO LIMITING NEEDED
                    if img_id in image_keypoints and image_keypoints[img_id]:
                        combined_keypoints = np.vstack(image_keypoints[img_id])
                    else:
                        combined_keypoints = None
                
                desc_h5.create_dataset(img_id, data=combined_descriptors, compression="gzip")
                
                # SAVE KEYPOINTS TOO
                if combined_keypoints is not None:
                    kp_h5.create_dataset(img_id, data=combined_keypoints, compression="gzip")
    
    # Cleanup
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Multi-scale DISK feature extraction complete.")
    print(f"Descriptors saved to {desc_h5_path}")
    print(f"Keypoints saved to {kp_h5_path}") 

    


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
                # zero keypoints; mirror behaviour of DISK extractor (skip image)
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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Feature extraction using DISK")
    parser.add_argument('--model', type = str, help = "Path to the model's .pth save file", default = MODEL_PATH)
    parser.add_argument('--df', type = str, help = "Path to the saved dataframe file", default = DATAFRAME_PATH)
    parser.add_argument('--output_dir', type = str, default = SAVE_TEST_DESCRIPTORS_PATH)

    args = parser.parse_args()
    
    img_paths = get_image_paths(args.df)
    extract_features(img_paths, args.model, args.output_dir)


