import os
import numpy as np
import cv2
import argparse
import sys
import pandas as pd
import torch
from constants import MODEL_PATH, DATAFRAME_PATH, DEVICE, SAVE_TEST_DESCRIPTORS_PATH, SAVE_TRAIN_DESCRIPTORS_PATH


def get_image_paths(df):

    

    #df = pd.read_csv(df_path)
    #if split == 'train':
    #    img_locations = df[df['original_split'] == 'train']
    #elif split == 'test':
    #    img_locations = df[df['original_split'] == 'test']
    #else:
    #    return None
    #print(len(img_locations))
    img_locations = (df['processed_path'] + "/" + df['image_id'] + '.jpg').tolist()
    
    print(img_locations[0])
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

    try:
        detect.extract(dataset, output_dir, model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        print("Feature extraction complete.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Feature extraction using DISK")
    parser.add_argument('--model', type = str, help = "Path to the model's .pth save file", default = MODEL_PATH)
    parser.add_argument('--df', type = str, help = "Path to the saved dataframe file", default = DATAFRAME_PATH)
    parser.add_argument('--output_dir', type = str, default = SAVE_TEST_DESCRIPTORS_PATH)

    args = parser.parse_args()
    
    img_paths = get_image_paths(args.df)
    extract_features(img_paths, args.model, args.output_dir)


