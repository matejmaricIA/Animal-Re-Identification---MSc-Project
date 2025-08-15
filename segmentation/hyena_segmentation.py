#!/usr/bin/env python3

"""
HyenaID2022-specific segmentation for spotted hyenas.

Challenges:
- Spotted coat pattern can create fragmented segmentation
- Often blend with savanna/grassland backgrounds
- Variable lighting conditions in wildlife photos
- Social animals - multiple hyenas in frame
- Similar coloring to dried grass and earth

Strategy:
- Use ISNet for robust animal detection
- Pattern-aware preprocessing to handle spotted coats
- Morphological operations to connect spotted regions
- Size and shape filtering for hyena-like bodies
"""

import cv2
import numpy as np
from typing import Optional

def enhance_hyena_features(image: np.ndarray) -> np.ndarray:
    """Enhance image to better distinguish hyena features."""
    # Convert to LAB color space for better control
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Enhance contrast in L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Slightly enhance A and B channels to bring out color differences
    lab[:, :, 1] = cv2.multiply(lab[:, :, 1], 1.1)
    lab[:, :, 2] = cv2.multiply(lab[:, :, 2], 1.1)
    lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
    
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Light edge enhancement to distinguish from background
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))
    enhanced = cv2.addWeighted(enhanced, 0.9, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.1, 0)
    
    return enhanced

def clean_spotted_mask(mask: np.ndarray) -> np.ndarray:
    """Clean up mask to handle hyena spotted patterns."""
    if mask is None or mask.size == 0:
        return mask
    
    # Remove small noise first
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # Connect nearby regions - important for spotted coats
    # Use larger kernel to connect spots that belong to same animal
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large)
    
    # Fill small holes within the hyena body
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill)
    
    # Find contours and filter by reasonable hyena characteristics
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cleaned
    
    # Filter contours by area and shape
    image_area = mask.shape[0] * mask.shape[1]
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Hyenas should be a reasonable size
        if area < image_area * 0.01:  # Too small
            continue
        if area > image_area * 0.7:   # Too large
            continue
            
        # Check aspect ratio - hyenas are roughly rectangular-ish
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Reasonable aspect ratio range for hyenas
        if 0.3 <= aspect_ratio <= 3.0:
            valid_contours.append(contour)
    
    if valid_contours:
        # Create mask from valid contours
        result = np.zeros_like(mask)
        cv2.fillPoly(result, valid_contours, 255)
        return result
    else:
        # If no valid contours, keep largest one
        if contours:
            largest = max(contours, key=cv2.contourArea)
            result = np.zeros_like(mask)
            cv2.fillPoly(result, [largest], 255)
            return result
        return cleaned

def segment(image: np.ndarray) -> Optional[np.ndarray]:
    """HyenaID2022 segmentation using ISNet with hyena-specific enhancements."""
    
    if image is None:
        return None
    
    try:
        # Enhanced preprocessing for hyenas
        enhanced = enhance_hyena_features(image)
        
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
        
        # Clean up spotted pattern
        cleaned_mask = clean_spotted_mask(binary_mask)
        
        # Apply mask to original image
        if cleaned_mask is not None:
            return cv2.bitwise_and(image, image, mask=cleaned_mask)
        else:
            return image
            
    except ImportError:
        print("rembg not available, using fallback hyena segmentation...")
        return fallback_hyena_segment(image)
    except Exception as e:
        print(f"ISNet segmentation failed: {e}, using fallback...")
        return fallback_hyena_segment(image)

def fallback_hyena_segment(image: np.ndarray) -> np.ndarray:
    """Fallback segmentation for hyenas using traditional CV methods."""
    # Enhanced preprocessing
    enhanced = enhance_hyena_features(image)
    
    # Convert to HSV for color-based segmentation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    
    # Define color range for hyena colors (tans, browns, grays)
    # Lower bound: darker browns/grays
    lower_hyena = np.array([8, 20, 30])
    # Upper bound: lighter tans/beiges
    upper_hyena = np.array([25, 180, 200])
    
    # Create color mask
    color_mask = cv2.inRange(hsv, lower_hyena, upper_hyena)
    
    # Also try to detect darker spots
    lower_spots = np.array([0, 10, 20])
    upper_spots = np.array([30, 150, 120])
    spots_mask = cv2.inRange(hsv, lower_spots, upper_spots)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(color_mask, spots_mask)
    
    # Clean up the combined mask
    final_mask = clean_spotted_mask(combined_mask)
    
    return cv2.bitwise_and(image, image, mask=final_mask)
