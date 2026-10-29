"""One call -> every feature for one image. This is also what the dashboard calls."""
import numpy as np
from .shape import shape_features
from .glcm import glcm_features, stat_features
from .hog_features import hog_features

GROUPS = ("shape", "stat", "glcm", "hog")
_NAMES = None


def extract_features(img, mask):
    """img: uint8 grayscale, mask: binary (same size). Returns an ordered dict of floats plus mask_ok.
    If the mask is empty/tiny, every feature is NaN and mask_ok = 0 (the classifier imputes these)."""
    mask = mask > 0
    if mask.sum() < 0.01 * mask.size:
        return {"mask_ok": 0.0, **{k: np.nan for k in feature_names()}}
    f = {"mask_ok": 1.0}
    for fn in (shape_features, stat_features, glcm_features, hog_features):
        f.update(fn(mask) if fn is shape_features else fn(img, mask))
    return f


def feature_names():
    global _NAMES
    if _NAMES is None:
        rng = np.random.default_rng(0)
        img = (rng.random((256, 256)) * 255).astype(np.uint8)
        m = np.zeros((256, 256), bool); m[60:200, 30:110] = True; m[60:200, 150:230] = True
        _NAMES = [k for k in extract_features(img, m) if k != "mask_ok"]
    return _NAMES


def groups_of(columns):
    return {g: [c for c in columns if c.startswith(g + "_")] for g in GROUPS}
