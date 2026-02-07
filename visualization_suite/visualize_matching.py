import torch
import cv2
import numpy as np
from pathlib import Path
from lightglue import DISK, LightGlue
import os

# Define paths
RAW_DIR = Path("data/elpephants/dataset/42")
SEG_DIR = Path("data/elpephants/segmented_dataset/42")
OUTPUT_DIR = Path("visualization_suite/output")
IMG_NAME_1 = "0487db74562a0743.jpg"
IMG_NAME_2 = "28660d243d483044.jpg"

INDICES = 'RANDOM' # 'RANDOM' or 'BEST'

def process_image_soft_mask_crop(raw_path, seg_path):
    # Load images
    raw_img = cv2.imread(str(raw_path))
    seg_img = cv2.imread(str(seg_path))
    
    if raw_img is None:
        raise FileNotFoundError(f"Raw image not found: {raw_path}")
    if seg_img is None:
        raise FileNotFoundError(f"Segmented image not found: {seg_path}")
        
    # Create mask from segmented image (non-black pixels)
    gray_seg = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_seg, 1, 255, cv2.THRESH_BINARY)
    
    # Find bounding box
    points = cv2.findNonZero(mask)
    if points is None:
        print(f"Warning: No mask found for {seg_path}, using raw image.")
        return raw_img, raw_img, np.array([0,0,0])
        
    x, y, w, h = cv2.boundingRect(points)
    
    # Pad crop slightly if possible to avoid edge cutoff
    pad = 10
    h_img, w_img = raw_img.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(w_img - x, w + 2*pad)
    h = min(h_img - y, h + 2*pad)
    
    # Crop
    raw_crop = raw_img[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]
    
    # Soft Mask
    # 1. Convert mask to float 0..1
    mask_float = mask_crop.astype(np.float32) / 255.0
    
    # 2. Blur the mask (Alpha channel)
    # Sigma 2.0
    alpha = cv2.GaussianBlur(mask_float, (0, 0), 2.0)
    alpha_3c = np.dstack([alpha]*3)
    
    # 3. Calculate Mean Background Color
    # Use the hard mask to select foreground pixels
    foreground_pixels = raw_crop[mask_crop > 0]
    if len(foreground_pixels) > 0:
        mean_bgr = np.mean(foreground_pixels, axis=0)
    else:
        mean_bgr = np.array([0, 0, 0], dtype=np.float32)
    
    # Create background image filled with mean color
    bg_img = np.full_like(raw_crop, mean_bgr, dtype=np.float32)
    
    # 4. Alpha Blend for Visualization
    # Output = Image * Alpha + Background * (1 - Alpha)
    raw_float = raw_crop.astype(np.float32)
    processed_img_soft = (raw_float * alpha_3c + bg_img * (1.0 - alpha_3c)).astype(np.uint8)
    
    # 5. Hard Mask for Matching (Black Background)
    # Matching should be done on the clean, hard-masked image
    processed_img_hard = cv2.bitwise_and(raw_crop, raw_crop, mask=mask_crop)
    
    return processed_img_soft, processed_img_hard, mean_bgr

def resize_and_pad(img, target_size, bg_color):
    """Resize image to fit in target_size while maintaining aspect ratio, and pad."""
    th, tw = target_size
    h, w = img.shape[:2]
    
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Create target image filled with bg_color
    canvas = np.full((th, tw, 3), bg_color, dtype=np.uint8)
    
    # Center it
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    
    canvas[dy:dy+nh, dx:dx+nw] = img_resized
    
    return canvas, scale, dx, dy

def numpy_to_tensor(img_np, device):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    # Prepare images
    raw1 = RAW_DIR / IMG_NAME_1
    seg1 = SEG_DIR / IMG_NAME_1
    raw2 = RAW_DIR / IMG_NAME_2
    seg2 = SEG_DIR / IMG_NAME_2

    print("Processing images (Soft Mask & Crop)...")
    img1_vis, img1_match, mean1 = process_image_soft_mask_crop(raw1, seg1)
    img2_vis, img2_match, mean2 = process_image_soft_mask_crop(raw2, seg2)
    
    # Resize and pad both images to a uniform size for the paper
    TARGET_SIZE = (640, 640)
    img1_canvas, scale1, dx1, dy1 = resize_and_pad(img1_vis, TARGET_SIZE, mean1)
    img2_canvas, scale2, dx2, dy2 = resize_and_pad(img2_vis, TARGET_SIZE, mean2)

    # Use Hard Masked images for Matching
    t_img1 = numpy_to_tensor(img1_match, device)
    t_img2 = numpy_to_tensor(img2_match, device)

    # Initialize models
    print("Initializing models...")
    extractor = DISK(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='disk').eval().to(device)

    # Extract & Match
    print("Matching...")
    with torch.inference_mode():
        feats0 = extractor.extract(t_img1)
        feats1 = extractor.extract(t_img2)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        
        matches = matches01['matches'][0] # (M, 2)
        scores = matches01['scores'][0]   # (M,)
        
        kpts0 = feats0['keypoints'][0]
        kpts1 = feats1['keypoints'][0]

        # Sort by scores
        sorted_indices = torch.argsort(scores, descending=True)
        
        top_k = 10
        top_indices = sorted_indices[:top_k]
        if INDICES == 'RANDOM':
            top_indices = np.random.choice(len(sorted_indices), top_k, replace=False).tolist()
        matches = matches[top_indices]
        
        m_kpts0 = kpts0[matches[:, 0]]
        m_kpts1 = kpts1[matches[:, 1]]
        
        m_kpts0_np = m_kpts0.cpu().numpy()
        m_kpts1_np = m_kpts1.cpu().numpy()

    print(f"Showing top {len(m_kpts0_np)} matches.")

    # Visualization - Use Resized Canvases
    tw, th = TARGET_SIZE
    vis_img = np.zeros((th, tw * 2, 3), dtype=np.uint8)
    
    # Place images
    vis_img[:, :tw] = img1_canvas
    vis_img[:, tw:tw*2] = img2_canvas
    
    # Draw matches
    # Lines: Green (0, 255, 0)
    # Dots: Red (0, 0, 255)
    
    color_line = (0, 255, 0)
    color_pt = (0, 0, 255)
    
    for i in range(len(m_kpts0_np)):
        # Adjust keypoints based on scale and padding
        px1 = int(m_kpts0_np[i, 0] * scale1 + dx1)
        py1 = int(m_kpts0_np[i, 1] * scale1 + dy1)
        
        px2 = int(m_kpts1_np[i, 0] * scale2 + dx2 + tw)
        py2 = int(m_kpts1_np[i, 1] * scale2 + dy2)
        
        pt1 = (px1, py1)
        pt2 = (px2, py2)
        
        # Thicker lines
        cv2.line(vis_img, pt1, pt2, color_line, 1, cv2.LINE_AA)
        
        # Bigger dots
        cv2.circle(vis_img, pt1, 4, color_pt, -1, cv2.LINE_AA)
        cv2.circle(vis_img, pt2, 4, color_pt, -1, cv2.LINE_AA)
        
        # White outline
        #cv2.circle(vis_img, pt1, 8, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.circle(vis_img, pt2, 8, (255, 255, 255), 1, cv2.LINE_AA)

    out_file = OUTPUT_DIR / "visualize_matches.png"
    cv2.imwrite(str(out_file), vis_img)
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    main()
