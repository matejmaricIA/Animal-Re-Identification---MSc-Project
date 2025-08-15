#!/usr/bin/env python3

"""
Giraffe-specific segmentation for savanna giraffes.

Challenges:
- Very tall animals that may extend beyond frame borders
- Distinctive pattern that can be confused with tree bark/shadows
- Long neck and legs create complex shapes
- Savanna backgrounds with similar brown/tan colors
- Multiple giraffes in same frame

Strategy:
- Use ISNet for robust shape detection
- Pattern-aware preprocessing to enhance giraffe spots
- Morphological operations to handle long necks and legs
- Size and aspect ratio filtering for giraffe-like shapes
"""

import cv2
import numpy as np
from typing import Optional

def enhance_giraffe_patterns(image: np.ndarray) -> np.ndarray:
    """Enhance giraffe spot patterns for better segmentation."""
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Enhance saturation to make brown spots more distinct
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Enhance value channel to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply light sharpening to enhance spot boundaries
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
    
    return enhanced

def post_process_giraffe_mask(mask: np.ndarray, min_aspect_ratio: float = 0.3) -> np.ndarray:
    """Post-process mask to handle giraffe-specific shape characteristics."""
    if mask is None or mask.size == 0:
        return mask
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Connect nearby regions (important for necks and legs)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    
    # Find contours and filter by giraffe-like characteristics
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cleaned
    
    valid_contours = []
    image_area = mask.shape[0] * mask.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area (giraffes should be reasonably large)
        if area < image_area * 0.02:  # Too small
            continue
        if area > image_area * 0.8:   # Too large (likely noise)
            continue
        
        # Check aspect ratio (giraffes are typically tall)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Giraffes should have reasonable aspect ratio (not too wide)
        if aspect_ratio <= 2.0:  # Allow for various poses
            valid_contours.append(contour)
    
    if valid_contours:
        # Create mask from valid contours
        result = np.zeros_like(mask)
        cv2.fillPoly(result, valid_contours, 255)
        return result
    else:
        # If no valid contours, return largest one
        if contours:
            largest = max(contours, key=cv2.contourArea)
            result = np.zeros_like(mask)
            cv2.fillPoly(result, [largest], 255)
            return result
        return cleaned

def segment(image: np.ndarray) -> Optional[np.ndarray]:
    """Giraffe segmentation using ISNet with giraffe-specific enhancements."""
    
    if image is None:
        return None
    
    try:
        # Enhanced preprocessing for giraffes
        enhanced = enhance_giraffe_patterns(image)
        
        # Try to import and use ISNet via rembg
        from rembg import remove, new_session
        
        # Create ISNet session for general object segmentation
        session = new_session('isnet-general-use')
        
        # Apply ISNet background removal on enhanced image
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        result = remove(rgb_enhanced, session=session)
        
        # Extract mask from alpha channel
        if len(result.shape) == 3 and result.shape[2] == 4:
            # RGBA result - extract alpha channel as mask
            alpha_mask = result[:, :, 3]
            # Convert to binary mask
            _, binary_mask = cv2.threshold(alpha_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            # Fallback if no alpha channel
            gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            _, binary_mask = cv2.threshold(gray_result, 127, 255, cv2.THRESH_BINARY)
        
        # Post-process with giraffe-specific rules
        processed_mask = post_process_giraffe_mask(binary_mask)
        
        # Apply mask to original image
        if processed_mask is not None:
            return cv2.bitwise_and(image, image, mask=processed_mask)
        else:
            return image
            
    except ImportError:
        print("rembg not available, using fallback giraffe segmentation...")
        return fallback_giraffe_segment(image)
    except Exception as e:
        print(f"ISNet segmentation failed: {e}, using fallback...")
        return fallback_giraffe_segment(image)

def fallback_giraffe_segment(image: np.ndarray) -> np.ndarray:
    """Fallback segmentation for giraffes using traditional CV methods."""
    # Enhanced preprocessing
    enhanced = enhance_giraffe_patterns(image)
    
    # Convert to HSV for color-based segmentation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    
    # Define color range for giraffe colors (browns, tans, oranges)
    # Lower bound: darker browns
    lower_giraffe = np.array([5, 40, 40])
    # Upper bound: lighter tans/yellows
    upper_giraffe = np.array([25, 255, 255])
    
    # Create color mask
    color_mask = cv2.inRange(hsv, lower_giraffe, upper_giraffe)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # Post-process the mask
    final_mask = post_process_giraffe_mask(color_mask)
    
    return cv2.bitwise_and(image, image, mask=final_mask)
