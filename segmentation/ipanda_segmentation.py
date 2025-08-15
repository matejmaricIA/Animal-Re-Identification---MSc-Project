#!/usr/bin/env python3

"""
IPanda50-specific segmentation for giant pandas.

Challenges:
- Black and white fur patterns can confuse standard segmentation
- Indoor/outdoor environments with varying backgrounds
- Pandas often blend with shadows due to black fur
- White fur can be confused with snow or bright backgrounds

Strategy:
- Use ISNet as primary method (good for complex black/white patterns)
- Enhanced preprocessing to better distinguish panda fur patterns
- Morphological operations to clean up fragmented regions
"""

import cv2
import numpy as np
from typing import Optional

def enhance_panda_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance contrast specifically for panda black/white patterns."""
    # Convert to LAB for better luminance control
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE to L channel to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Enhance the contrast between black and white regions
    # Use sigmoid-like curve to make darks darker and lights lighter
    l_enhanced = l_enhanced.astype(np.float32)
    l_enhanced = 255 * (1 / (1 + np.exp(-0.1 * (l_enhanced - 127))))
    l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
    
    lab[:, :, 0] = l_enhanced
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def clean_panda_mask(mask: np.ndarray, min_area_ratio: float = 0.01) -> np.ndarray:
    """Clean up segmentation mask specifically for pandas."""
    if mask is None or mask.size == 0:
        return mask
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes in panda body
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small components, keep largest
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned
    
    # Filter by area
    image_area = mask.shape[0] * mask.shape[1]
    min_area = int(image_area * min_area_ratio)
    
    large_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if large_contours:
        # Create mask from largest components
        result = np.zeros_like(mask)
        cv2.fillPoly(result, large_contours, 255)
        return result
    else:
        return cleaned

def segment(image: np.ndarray) -> Optional[np.ndarray]:
    """IPanda50 segmentation using ISNet with panda-specific enhancements."""
    
    if image is None:
        return None
    
    try:
        # Enhanced preprocessing for pandas
        enhanced = enhance_panda_contrast(image)
        
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
        
        # Clean up the mask with panda-specific processing
        cleaned_mask = clean_panda_mask(binary_mask, min_area_ratio=0.02)
        
        # Apply mask to original image
        if cleaned_mask is not None:
            return cv2.bitwise_and(image, image, mask=cleaned_mask)
        else:
            return image
            
    except ImportError:
        print("rembg not available, using fallback panda segmentation...")
        return fallback_panda_segment(image)
    except Exception as e:
        print(f"ISNet segmentation failed: {e}, using fallback...")
        return fallback_panda_segment(image)

def fallback_panda_segment(image: np.ndarray) -> np.ndarray:
    """Fallback segmentation for pandas using traditional CV methods."""
    # Enhanced preprocessing
    enhanced = enhance_panda_contrast(image)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find panda boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create connected regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours and create mask from largest region
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour (likely the panda)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        # Clean up the mask
        mask = clean_panda_mask(mask)
        
        return cv2.bitwise_and(image, image, mask=mask)
    else:
        # If no contours found, return original image
        return image
