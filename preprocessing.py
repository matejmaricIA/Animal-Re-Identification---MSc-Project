from multiprocessing import Pool, cpu_count
import os
import cv2
import numpy as np
from rembg import remove, new_session
import argparse
from constants import *
from tqdm import tqdm

if SEGMENTATION_MODEL_TYPE == 'isnet':
    session = new_session(ISNET_MODEL_NAME)
elif SEGMENTATION_MODEL_TYPE == 'sam':
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import torch
    device = 'cuda' if DEVICE.upper() == 'GPU' and torch.cuda.is_available() else 'cpu'
    _sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device)
    mask_generator = SamAutomaticMaskGenerator(_sam)
else:
    session = None

def mantiuk_tone_mapping(image):
    
    #image = cv2.imread(os.path.join('./data/ATRW', image_path))
    
    tonemapMantiuk = cv2.createTonemapMantiuk(scale = 0.7, saturation = 0.7)
    image = image.astype(np.float32) / 255.0
    
    mantiuk_image = tonemapMantiuk.process(image)
    mantiuk_image = np.clip(mantiuk_image * 255, 0,  255).astype(np.uint8)
    
    return mantiuk_image

def background_removal(image):
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
        from sam2 import Sam2
        sam2 = Sam2(SAM2_CHECKPOINT_PATH)
        masks = sam2.predict(image)
        if not masks:
            return image
        largest_mask = max(masks, key=lambda m: m['area'])
        mask = largest_mask['segmentation'].astype(np.uint8) * 255
        return cv2.bitwise_and(image, image, mask=mask)
    else:
        print(f"Unsupported segmentation model type: {SEGMENTATION_MODEL_TYPE}")
        return image
def process_image(row, output_dir, use_mantiuk, dataset_name, remove_background):
    """ Process an image by applying Mantiuk tone mapping and background removal."""

    #image_path = os.path.join(f'./data/{dataset_name}', row['path'])
    image_path = os.path.join(f'{WILD_DATASET_PATH}', row['path'])
    identity = row['identity']
    #split = row['original_split']
    id = str(row['image_id'])
    save_dir = os.path.join(output_dir, str(identity))
    os.makedirs(save_dir, exist_ok = True)

    image = cv2.imread(image_path)
    if use_mantiuk:
        image = mantiuk_tone_mapping(image)
    
    if remove_background:
        masked_image = background_removal(image)
    else:
        masked_image = image
    cv2.imwrite(os.path.join(save_dir, f'{id}.jpg'), masked_image)

    return save_dir


def preprocess_dataset(df, output_dir, dataset_name, use_mantiuk = True, remove_background = True):
    """
    Preprocess the dataset by applying Mantiuk tone mapping
    and background removal using SAM and ISNet."""
    
    args = [(row, output_dir, use_mantiuk, dataset_name, remove_background) for _, row in df.iterrows()]
    
    # Process sequentially instead of using multiprocessing
    processed_paths = []
    #index = 0
    for arg in tqdm(args, desc = "Preprocessing images", unit = "image"):
        processed_path = process_image(*arg)
        processed_paths.append(processed_path)
        #print(f"Processed image {index}/{len(args)}")
        #index += 1
    df['processed_path'] = processed_paths
    return df
        
def preprocess_inference(image_paths, use_mantiuk=False, remove_background=False):
    save_dir = os.path.join(TMP, 'segmented')
    os.makedirs(save_dir, exist_ok = True)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if use_mantiuk:
            image = mantiuk_tone_mapping(image)
        
        if remove_background:
            masked_image = background_removal(image)
        else:
            masked_image = image

        cv2.imwrite(os.path.join(save_dir, f'{os.path.basename(image_path)}'), masked_image)

    return save_dir

        
if __name__ == '__main__':
    print('...')
    