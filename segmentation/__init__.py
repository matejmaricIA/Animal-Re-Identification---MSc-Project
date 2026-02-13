"""Dataset-specific segmentation utilities (Grounded SAM2)."""
from typing import Optional
import numpy as np


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_RAW_PROMPTS = {
    "atrw": "tiger",
    "elpephants": "elephant",
    "seastarreid2023": "sea star . starfish",
    "cowdataset": "cow . cattle",
    "amvrakikosturtles": "turtle",
    "polarbearvidid": "bear . polar bear",
    "wildboar": "wild boar . boar",
    "roedeer": "roedeer . deer",
    "atrw_fewshot": "tiger",
    "elpephants_fewshot": "elephant",
    "seastarreid2023_fewshot": "sea star . starfish",
    "cowdataset_fewshot": "cow . cattle",
    "amvrakikosturtles_fewshot": "turtle",
    "polarbearvidid_fewshot": "bear . polar bear",
    "wildboar_fewshot": "wild boar . boar",
    "roedeer_fewshot": "roedeer . deer",
}

_PROMPTS = {_normalize_name(name): prompt for name, prompt in _RAW_PROMPTS.items()}


def has_segmenter(name: str) -> bool:
    return _normalize_name(name) in _PROMPTS


def get_prompt(name: str) -> Optional[str]:
    return _PROMPTS.get(_normalize_name(name))


def segment_image(name: str, image: np.ndarray, **kwargs) -> Optional[np.ndarray]:
    prompt = get_prompt(name)
    if prompt is None:
        return None
    from .grounded_sam2 import segment
    return segment(image, prompt=prompt, **kwargs)


def segment_dataset(
    df,
    output_dir: str,
    dataset_name: str,
    use_mantiuk: bool = True,
):
    """Preprocess and segment an entire dataset using preprocessing pipeline."""
    from preprocessing import preprocess_dataset
    return preprocess_dataset(
        df,
        output_dir,
        dataset_name,
        use_mantiuk=use_mantiuk,
        remove_background=True,
    )
