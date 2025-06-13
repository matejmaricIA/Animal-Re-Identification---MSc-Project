import os
import numpy as np
import cv2
import argparse
import sys
import pandas as pd
import torch

# Create HDF5 file for storing descriptors
import h5py

# For keynet, hardnet, and affnet
import kornia
from kornia.feature import KeyNet, HardNet
from kornia.geometry import warp_affine

from constants import MODEL_PATH, DATAFRAME_PATH, DEVICE, SAVE_TEST_DESCRIPTORS_PATH, SAVE_TRAIN_DESCRIPTORS_PATH, PATH_SIZE


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
    
def extract_features_keynet_hardnet(image_paths, output_dir, max_keypoints = 8000):
    """Extract features using Key.Net + AffNet + HardNet"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    keynet = KeyNet(pretrained=True).to(device).eval()
    #affnet = AffNet(pretrained=True).to(device).eval()
    hardnet = HardNet(pretrained=True).to(device).eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    descriptors_file = os.path.join(output_dir, 'descriptors.h5')
    
    with h5py.File(descriptors_file, 'w') as f:
        for img_path in image_paths:
            try:
                # Load and preprocess image
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                
                # Convert to tensor and normalize
                image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
                image_tensor = image_tensor.to(device)
                
                with torch.no_grad():
                    # Detect keypoints
                    keypoints = keynet(image_tensor)
                    
                    # Limit number of keypoints
                    if keypoints.shape[1] > max_keypoints:
                        # Sort by response and take top keypoints
                        scores = keypoints[0, :, 2]  # Response scores
                        _, indices = torch.topk(scores, max_keypoints)
                        keypoints = keypoints[:, indices, :]
                    
                    if keypoints.shape[1] == 0:
                        print(f"Warning: No keypoints detected in {img_path}")
                        # Store empty array
                        image_id = os.path.splitext(os.path.basename(img_path))[0]
                        f.create_dataset(image_id, data=np.array([]).reshape(0, 128))
                        continue
                    
                    # Extract patches around keypoints
                    patches = kornia.feature.extract_patches_from_pyramid(
                        image_tensor, keypoints, 32  # patch size
                    )
                    
                    # Estimate affine shapes
                    affine_shapes = affnet(patches)
                    
                    # Apply affine transformation to patches
                    warped_patches = []
                    for i in range(patches.shape[0]):
                        patch = patches[i:i+1]
                        affine_matrix = affine_shapes[i:i+1]
                        warped_patch = warp_affine(patch, affine_matrix, (32, 32))
                        warped_patches.append(warped_patch)
                    
                    if warped_patches:
                        warped_patches = torch.cat(warped_patches, dim=0)
                        
                        # Extract descriptors
                        descriptors = hardnet(warped_patches)
                        descriptors = descriptors.cpu().numpy()
                    else:
                        descriptors = np.array([]).reshape(0, 128)
                
                # Store descriptors
                image_id = os.path.splitext(os.path.basename(img_path))[0]
                f.create_dataset(image_id, data=descriptors)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Store empty array for failed images
                image_id = os.path.splitext(os.path.basename(img_path))[0]
                f.create_dataset(image_id, data=np.array([]).reshape(0, 128))
    
    print("Key.Net + HardNet feature extraction complete.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Feature extraction using DISK")
    parser.add_argument('--model', type = str, help = "Path to the model's .pth save file", default = MODEL_PATH)
    parser.add_argument('--df', type = str, help = "Path to the saved dataframe file", default = DATAFRAME_PATH)
    parser.add_argument('--output_dir', type = str, default = SAVE_TEST_DESCRIPTORS_PATH)

    args = parser.parse_args()
    
    img_paths = get_image_paths(args.df)
    extract_features(img_paths, args.model, args.output_dir)


