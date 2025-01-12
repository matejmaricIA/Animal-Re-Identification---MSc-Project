from multiprocessing import Pool
import os
import cv2
import numpy as np


def mantiuk_tone_mapping(image):
    tonemapMantiuk = cv2.createTonemapMantiuk(scale = 0.7, saturation = 0.7)
    image = image.astype(np.float32) / 255.0
    
    mantiuk_image = tonemapMantiuk.process(image)
    mantiuk_image = np.clip(mantiuk_image * 255, 0,  255).astype(np.uint8)
    
    return mantiuk_image

def background_removal(image, save_path):
    """Remove the background using SAM and ISNet."""
    pass

def preprocess_dataset(input_dir, output_dir_segmented, use_mantiuk = True):
    """
    Preprocess the dataset by applying Mantiuk tone mapping
    and background removal using SAM and ISNet."""
    pass