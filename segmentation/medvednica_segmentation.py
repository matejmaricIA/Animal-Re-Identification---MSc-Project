#!/usr/bin/env python3

"""
MedvednicaDS-specific segmentation for wild boar and deer from trail cameras.

Challenges:
- Trail camera images with varying lighting (day/night, IR flash)
- Dense forest backgrounds with similar brown/tan colors
- Animals often partially obscured by vegetation
- Motion blur from moving animals
- Multiple species (deer and wild boar) with different characteristics

Strategy:
- Use ISNet as primary method for robust animal detection
- Adaptive preprocessing for trail camera image characteristics
- Species-aware post-processing for deer vs wild boar shapes
- Motion blur handling and low-light enhancement
"""

import cv2
import numpy as np
from typing import Optional

def enhance_trail_camera_image(image: np.ndarray) -> np.ndarray:
    """Enhance trail camera images for better segmentation."""
    # Check if image appears to be night/IR (low saturation, bluish tint)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_value = np.mean(hsv[:, :, 2])
    
    is_low_light = avg_saturation < 50 or avg_value < 80
    
    if is_low_light:
        # Night/IR image enhancement
        enhanced = enhance_night_image(image)
    else:
        # Daylight image enhancement
        enhanced = enhance_daylight_image(image)
    
    return enhanced

def enhance_night_image(image: np.ndarray) -> np.ndarray:
    """Enhance night/IR trail camera images."""
    # Convert to LAB for better control over luminance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Aggressive contrast enhancement for low-light conditions
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Reduce noise common in night images
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Sharpen to counteract blur
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    enhanced = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
    
    return enhanced

def enhance_daylight_image(image: np.ndarray) -> np.ndarray:
    """Enhance daylight trail camera images."""
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Enhance saturation to better distinguish animals from vegetation
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Mild contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Light denoising
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return enhanced

def clean_forest_animal_mask(mask: np.ndarray) -> np.ndarray:
    """Clean up mask for forest animals (deer and wild boar)."""
    if mask is None or mask.size == 0:
        return mask
    
    # Remove small noise (twigs, leaves, etc.)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    # Connect fragmented parts of animals (legs, ears that might be separated)
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect)
    
    # Fill holes within animal bodies
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill)
    
    # Filter contours by characteristics appropriate for deer and wild boar
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cleaned
    
    image_area = mask.shape[0] * mask.shape[1]
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by reasonable animal size
        if area < image_area * 0.02:  # Too small (probably debris)
            continue
        if area > image_area * 0.8:   # Too large (probably background noise)
            continue
        
        # Check aspect ratio - both deer and wild boar have reasonable proportions
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Accept a wide range of aspect ratios for different poses and species
        if 0.2 <= aspect_ratio <= 4.0:
            valid_contours.append(contour)
    
    if valid_contours:
        # Create mask from valid contours
        result = np.zeros_like(mask)
        cv2.fillPoly(result, valid_contours, 255)
        return result
    else:
        # If no valid contours found, keep largest
        if contours:
            largest = max(contours, key=cv2.contourArea)
            result = np.zeros_like(mask)
            cv2.fillPoly(result, [largest], 255)
            return result
        return cleaned

def segment(image: np.ndarray) -> Optional[np.ndarray]:
    """MedvednicaDS segmentation for deer and wild boar using ISNet."""
    
    if image is None:
        return None
    
    try:
        # Enhanced preprocessing for trail camera images
        enhanced = enhance_trail_camera_image(image)
        
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
        
        # Clean up with forest-specific processing
        cleaned_mask = clean_forest_animal_mask(binary_mask)
        
        # Apply mask to original image
        if cleaned_mask is not None:
            return cv2.bitwise_and(image, image, mask=cleaned_mask)
        else:
            return image
            
    except ImportError:
        print("rembg not available, using fallback forest animal segmentation...")
        return fallback_forest_segment(image)
    except Exception as e:
        print(f"ISNet segmentation failed: {e}, using fallback...")
        return fallback_forest_segment(image)

def fallback_forest_segment(image: np.ndarray) -> np.ndarray:
    """Fallback segmentation for forest animals using traditional CV methods."""
    # Enhanced preprocessing
    enhanced = enhance_trail_camera_image(image)
    
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for deer and wild boar
    # Deer: browns, tans, reddish-browns
    lower_deer = np.array([8, 30, 40])
    upper_deer = np.array([25, 255, 200])
    
    # Wild boar: darker browns, grays, blacks
    lower_boar = np.array([0, 10, 20])
    upper_boar = np.array([30, 180, 120])
    
    # Create masks for both species
    deer_mask = cv2.inRange(hsv, lower_deer, upper_deer)
    boar_mask = cv2.inRange(hsv, lower_boar, upper_boar)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(deer_mask, boar_mask)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Final cleanup
    final_mask = clean_forest_animal_mask(combined_mask)
    
    return cv2.bitwise_and(image, image, mask=final_mask)
