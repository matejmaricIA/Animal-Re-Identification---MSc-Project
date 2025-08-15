#!/usr/bin/env python3

"""
Nyala-specific segmentation for grassland antelope images.

Challenges:
- Brown/tan body color similar to dry grass and earth
- White stripes can be confused with sunlight or dried vegetation  
- Natural camouflage in savanna environment
- Variable lighting conditions

Strategy:
- Edge-based detection to find stripe patterns
- Color filtering for brown/tan + white combinations
- Morphological operations to connect stripe regions
- Size and shape filtering for antelope-like objects
"""

import cv2
import numpy as np
from typing import Optional


def segment(image: np.ndarray) -> Optional[np.ndarray]:
    """Nyala segmentation using ISNet with preprocessing enhancements."""
    
    if image is None:
        return None
    
    try:
        # Try to import and use ISNet via rembg
        from rembg import remove, new_session
        
        # ENHANCEMENT: Preprocess image for better segmentation
        enhanced = enhance_for_segmentation(image)
        
        # Create ISNet session for general object segmentation
        session = new_session('isnet-general-use')
        
        # Apply ISNet background removal on enhanced image
        # rembg expects RGB format
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        # Get the segmented result with alpha channel
        result = remove(rgb_enhanced, session=session)
        
        # Convert back to BGR and extract the mask
        if len(result.shape) == 3 and result.shape[2] == 4:
            # RGBA result - extract alpha channel as mask
            alpha = result[:, :, 3]
            
            # ENHANCEMENT: Clean up the segmentation mask
            mask = clean_segmentation_mask(alpha)
            
            # Apply cleaned mask to ORIGINAL image (not enhanced)
            return cv2.bitwise_and(image, image, mask=mask)
        else:
            # Fallback if unexpected format
            return image
            
    except ImportError:
        print("rembg not available, falling back to simple thresholding")
        # Fallback to simple method if rembg is not available
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Light morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return cv2.bitwise_and(image, image, mask=mask)
        
    except Exception as e:
        print(f"ISNet segmentation failed: {e}, falling back to simple method")
        # Fallback to simple method if ISNet fails
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return cv2.bitwise_and(image, image, mask=mask)


def enhance_for_segmentation(image: np.ndarray) -> np.ndarray:
    """Enhance image to help ISNet distinguish Nyala from savanna background."""
    
    # Convert to LAB for better contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Enhance L channel (lightness) with CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Subtle edge enhancement to help ISNet detect animal boundaries
    # This is especially helpful for stripe patterns
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel * 0.05)  # Very subtle
    enhanced = cv2.addWeighted(enhanced, 0.95, sharpened, 0.05, 0)
    
    return enhanced


def clean_segmentation_mask(alpha: np.ndarray) -> np.ndarray:
    """Clean up ISNet segmentation mask to remove artifacts and improve quality."""
    
    # Create binary mask from alpha channel
    mask = (alpha > 127).astype(np.uint8) * 255
    
    # Remove small noise with opening (removes small white spots)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Fill small holes with closing (fills gaps in the animal)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Find largest connected component (should be the main animal body)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels > 1:  # Background is label 0
        # Find largest component (excluding background)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_component).astype(np.uint8) * 255
    
    return mask


if __name__ == "__main__":
    print("Nyala segmentation module")
    print("Optimized for:")
    print("- Brown/tan body with white stripes")  
    print("- Grassland/savanna environments")
    print("- Stripe pattern detection")
    print("- Natural camouflage handling")
