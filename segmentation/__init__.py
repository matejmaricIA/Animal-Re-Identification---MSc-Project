"""Dataset-specific segmentation utilities."""
from typing import Callable, Dict, Optional
import cv2
import numpy as np
from .beluga_segmentation import segment as beluga_segment
from .nyala_segmentation import segment as nyala_segment
from .ipanda_segmentation import segment as ipanda_segment
from .giraffe_segmentation import segment as giraffe_segment
from .hyena_segmentation import segment as hyena_segment
from .medvednica_segmentation import segment as medvednica_segment


def _threshold_segment(image: np.ndarray) -> np.ndarray:
    """Fallback simple threshold-based segmentation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_and(image, image, mask=mask)

# Registry of dataset name -> segmentation function
_SEGMENTERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "belugaid": beluga_segment,
    "atrw": _threshold_segment,
    "ipanda50": ipanda_segment,
    "nyaladata": nyala_segment,
    "hyenaid2022": hyena_segment,
    "giraffes": giraffe_segment,
    "cowdataset": _threshold_segment,
    "medvednicads": medvednica_segment,
}


def has_segmenter(name: str) -> bool:
    return name.lower() in _SEGMENTERS


def get_segmenter(name: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    return _SEGMENTERS.get(name.lower())


def segment_image(name: str, image: np.ndarray) -> Optional[np.ndarray]:
    seg_fn = get_segmenter(name)
    if seg_fn is None:
        return None
    return seg_fn(image)


def segment_dataset(df, output_dir: str, dataset_name: str, use_mantiuk: bool = True):
    """Preprocess and segment an entire dataset using preprocessing pipeline."""
    from preprocessing import preprocess_dataset  # lazy import to avoid circular
    return preprocess_dataset(
        df,
        output_dir,
        dataset_name,
        use_mantiuk=use_mantiuk,
        remove_background=True,
    )