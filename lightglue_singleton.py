from functools import lru_cache
from lightglue import LightGlue
import torch

@lru_cache(maxsize=None)          # one instance per (features, device) tuple
def get_lightglue(features='disk'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return LightGlue(features=features).to(device).eval()