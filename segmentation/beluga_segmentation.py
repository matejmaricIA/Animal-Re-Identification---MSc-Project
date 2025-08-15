import cv2
import numpy as np

def segment(image: np.ndarray) -> np.ndarray:
    """GrabCut-based beluga segmentation with intelligent initialization.
    
    This approach uses GrabCut algorithm with automatic foreground/background
    estimation based on brightness and position.
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Create initial mask for GrabCut
    mask = np.zeros((h, w), np.uint8)
    
    # Stage 1: Create rough foreground/background estimation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply mild blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Stage 2: Find bright areas (potential beluga)
    # Use adaptive threshold to handle varying lighting
    _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Stage 3: Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Stage 4: Find the largest bright region (likely the whale)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to simple thresholding if no contours found
        return cv2.bitwise_and(image, image, mask=bright_mask)
    
    # Keep only reasonably sized contours
    min_area = h * w * 0.01  # At least 1% of image
    max_area = h * w * 0.7   # At most 70% of image
    
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        # If no valid contours, use the largest one
        largest_contour = max(contours, key=cv2.contourArea)
        valid_contours = [largest_contour]
    
    # Stage 5: Set up GrabCut masks
    # Everything starts as "maybe background"
    mask[:] = cv2.GC_PR_BGD
    
    # Set obvious background (edges of image and very dark areas)
    border_width = min(h, w) // 20  # Small border
    mask[:border_width, :] = cv2.GC_BGD  # Top
    mask[-border_width:, :] = cv2.GC_BGD  # Bottom  
    mask[:, :border_width] = cv2.GC_BGD  # Left
    mask[:, -border_width:] = cv2.GC_BGD  # Right
    
    # Very dark areas are definitely background
    dark_threshold = np.percentile(gray, 15)  # Bottom 15% of brightness
    mask[gray < dark_threshold] = cv2.GC_BGD
    
    # Stage 6: Mark bright areas as probable foreground
    for contour in valid_contours:
        # Fill the contour area as probable foreground
        cv2.fillPoly(mask, [contour], cv2.GC_PR_FGD)
        
        # The center of the contour is definitely foreground
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Mark a small area around center as definite foreground
            center_size = min(h, w) // 30
            y1, y2 = max(0, cy - center_size), min(h, cy + center_size)
            x1, x2 = max(0, cx - center_size), min(w, cx + center_size)
            mask[y1:y2, x1:x2] = cv2.GC_FGD
    
    # Stage 7: Apply GrabCut
    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Run GrabCut for 5 iterations
        mask_gc, _, _ = cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        
        # Create final mask (foreground + probable foreground)
        final_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
    except Exception as e:
        print(f"GrabCut failed: {e}, using fallback")
        # Fallback to the initial bright mask
        final_mask = bright_mask
    
    # Stage 8: Final cleanup
    if np.sum(final_mask) < (h * w * 0.005):  # If mask is too small
        print("Result too small, using fallback mask")
        final_mask = bright_mask
    
    # Light morphological cleanup
    kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_cleanup)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_cleanup)
    
    return cv2.bitwise_and(image, image, mask=final_mask)