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
                                    num_features= MAX_KEYPOINTS,
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
            img_id, img_path = _parse_image_item(image_paths[img_idx])

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
    num_features = MAX_KEYPOINTS,
    batch_size = 16,
    num_workers = 0,
    use_half = True,
    image_max_size = 1024,
    memory_debug = True,
    skip_on_oom = True,
    save_failed_list = True,
    cpu_fallback = False 
):
    """Extract KeyNet+AffNet+HardNet features.

    """
    import gc
    import json
    from datetime import datetime
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lfeat = KeyNetAffNetHardNet(num_features=num_features).to(device).eval()

    if use_half and device.type == "cuda":
        lfeat = lfeat.half()
        
    # Resize to cap the long edge (speed/memory); later we rescale keypoints back
    # to original coordinates so GV thresholds remain meaningful.
    dataset = ImageDataset(image_paths, max_size=image_max_size)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x,
    )
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    desc_h5_path = Path(output_dir) / "descriptors.h5"
    kp_h5_path = Path(output_dir) / "keypoints.h5"
    failed_images_path = Path(output_dir) / "failed_images.json"
    
    dtype = torch.float16 if use_half and device.type == "cuda" else torch.float32
    
    # Track statistics
    failed_images = []
    processed_count = 0
    skipped_count = 0
    
    # Optional: Print initial memory usage
    if memory_debug and torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Device: {device}, dtype: {dtype}, half precision: {use_half}")
    
    # CPU fallback model (only create if needed)
    cpu_model = None
    
    with h5py.File(desc_h5_path, "w") as desc_h5, h5py.File(kp_h5_path, "w") as kp_h5:
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc="KeyNetAffNetHardNet")):
            for img_id, tens, (scale_x, scale_y) in batch:
                # Initialize variables for cleanup
                tens_gpu = None
                lafs = None
                descs = None
                desc_np = None
                keypoints_np = None
                
                try:
                    # Move tensor to device
                    tens_gpu = tens.to(device, dtype=dtype)
                    
                    # Perform inference
                    with torch.inference_mode():
                        lafs, _, descs = lfeat(tens_gpu.unsqueeze(0))
                    
                    # Convert to numpy immediately and move to CPU
                    desc_np = descs.squeeze(0).cpu().numpy().astype(np.float32)
                    keypoints_np = lafs[0, :, :, 2].cpu().numpy().astype(np.float32)

                    # Map keypoints back to original image coordinates.
                    if scale_x != 1.0 or scale_y != 1.0:
                        keypoints_np[:, 0] /= np.float32(scale_x)
                        keypoints_np[:, 1] /= np.float32(scale_y)
                    
                    # Skip if no keypoints detected
                    if desc_np.shape[0] == 0:
                        if memory_debug:
                            print(f"No keypoints detected for {img_id}")
                        skipped_count += 1
                    else:
                        # Save to HDF5
                        desc_h5.create_dataset(img_id, data=desc_np, compression="gzip")
                        kp_h5.create_dataset(img_id, data=keypoints_np, compression="gzip")
                        processed_count += 1
                    
                except torch.cuda.OutOfMemoryError as e:
                    error_msg = f"CUDA OOM: {str(e)}"
                    if memory_debug:
                        print(f"CUDA OOM error for {img_id}: {error_msg}")
                    
                    # Try CPU fallback if enabled
                    if cpu_fallback and skip_on_oom:
                        try:
                            if memory_debug:
                                print(f"Attempting CPU fallback for {img_id}")
                            
                            # Create CPU model if not exists
                            if cpu_model is None:
                                cpu_model = KeyNetAffNetHardNet(num_features=num_features).to("cpu").eval()
                            
                            # Process on CPU
                            tens_cpu = tens.to("cpu", dtype=torch.float32)
                            with torch.inference_mode():
                                lafs_cpu, _, descs_cpu = cpu_model(tens_cpu.unsqueeze(0))
                            
                            desc_np = descs_cpu.squeeze(0).cpu().numpy().astype(np.float32)
                            keypoints_np = lafs_cpu[0, :, :, 2].cpu().numpy().astype(np.float32)

                            # Map keypoints back to original image coordinates.
                            if scale_x != 1.0 or scale_y != 1.0:
                                keypoints_np[:, 0] /= np.float32(scale_x)
                                keypoints_np[:, 1] /= np.float32(scale_y)
                            
                            if desc_np.shape[0] > 0:
                                desc_h5.create_dataset(img_id, data=desc_np, compression="gzip")
                                kp_h5.create_dataset(img_id, data=keypoints_np, compression="gzip")
                                processed_count += 1
                                if memory_debug:
                                    print(f"Successfully processed {img_id} on CPU")
                            else:
                                skipped_count += 1
                                
                        except Exception as cpu_e:
                            failed_images.append({
                                "image_id": img_id,
                                "error_type": "CUDA_OOM_CPU_FALLBACK_FAILED",
                                "error_message": f"GPU: {error_msg}, CPU: {str(cpu_e)}",
                                "timestamp": datetime.now().isoformat()
                            })
                            if memory_debug:
                                print(f"CPU fallback also failed for {img_id}: {str(cpu_e)}")
                    else:
                        if skip_on_oom:
                            failed_images.append({
                                "image_id": img_id,
                                "error_type": "CUDA_OOM",
                                "error_message": error_msg,
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            raise e
                    
                    # Force memory cleanup after OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except RuntimeError as e:
                    error_msg = f"Runtime error: {str(e)}"
                    if memory_debug:
                        print(f"Runtime error for {img_id}: {error_msg}")
                    
                    if skip_on_oom:
                        failed_images.append({
                            "image_id": img_id,
                            "error_type": "RUNTIME_ERROR",
                            "error_message": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        raise e
                        
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    if memory_debug:
                        print(f"Unexpected error for {img_id}: {error_msg}")
                    
                    if skip_on_oom:
                        failed_images.append({
                            "image_id": img_id,
                            "error_type": "UNEXPECTED_ERROR",
                            "error_message": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        raise e
                
                finally:
                    # Aggressive cleanup - safely delete variables
                    variables_to_delete = ['tens_gpu', 'lafs', 'descs', 'desc_np', 'keypoints_np']
                    for var_name in variables_to_delete:
                        if var_name in locals() and locals()[var_name] is not None:
                            del locals()[var_name]
                    
                    # Memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Optional: Monitor memory usage periodically
                if memory_debug and torch.cuda.is_available() and batch_idx % 20 == 0:
                    print(f"Batch {batch_idx} - GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    print(f"Processed: {processed_count}, Skipped: {skipped_count}, Failed: {len(failed_images)}")
            
            # Additional cleanup after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Save failed images list
    if save_failed_list and failed_images:
        with open(failed_images_path, 'w') as f:
            json.dump(failed_images, f, indent=2)
        print(f"Failed images list saved to {failed_images_path}")
    
    # Print final statistics
    total_attempted = processed_count + skipped_count + len(failed_images)
    print(f"\n=== Feature Extraction Summary ===")
    print(f"Total images attempted: {total_attempted}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (no keypoints): {skipped_count}")
    print(f"Failed (errors): {len(failed_images)}")
    print(f"Success rate: {(processed_count/total_attempted)*100:.1f}%")
    
    if failed_images:
        print(f"\nFailed images breakdown:")
        error_types = {}
        for fail in failed_images:
            error_type = fail['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count} images")
    
    # Optional: Print final memory usage
    if memory_debug and torch.cuda.is_available():
        print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                
    print(f"\nKeyNet+AffNet+HardNet features saved to {desc_h5_path}")
    print(f"KeyNet+AffNet+HardNet keypoints saved to {kp_h5_path}")
    
    #return {
    #    'processed': processed_count,
    #    'skipped': skipped_count,
    #    'failed': len(failed_images),
    #    'failed_list': failed_images
    #}



def extract_features_lightglue(
    image_paths,
    output_dir,
    feature_type = "aliked",
    max_keypoints = MAX_KEYPOINTS,
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
