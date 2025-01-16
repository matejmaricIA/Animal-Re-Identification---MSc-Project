import os
import numpy as np
import cv2
import argparse
import sys
import pandas as pd
import torch

MODEL_PATH = './disk/depth-save.pth'
DATAFRAME_PATH = './data/processed_metadata.csv'
DEVICE = 'CPU'
SAVE_DESCRIPTORS_PATH = './data/feature_descriptors/'

def get_image_paths(df_path, split = 'train'):

    

    df = pd.read_csv(df_path)
    if split == 'train':
        img_locations = df[df['original_split'] == 'train']
    elif split == 'test':
        img_locations = df[df['original_split'] == 'test']
    else:
        return None
    #print(len(img_locations))
    img_locations = (img_locations['processed_path'] + "/" + img_locations['image_id'] + '.jpg').tolist()
    
    print(img_locations[0])
    return img_locations


def process_images(image_paths, model_path):
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
        described_samples = detect.extract(dataset, SAVE_DESCRIPTORS_PATH, model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        print("Feature extraction complete.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Feature extraction using DISK")
    parser.add_argument('--model', type = str, help = "Path to the model's .pth save file", default = MODEL_PATH)
    parser.add_argument('--df', type = str, help = "Path to the saved dataframe file", default = DATAFRAME_PATH)

    args = parser.parse_args()
    
    img_paths = get_image_paths(args.df, 'train')
    process_images(img_paths, args.model)


