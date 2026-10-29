"""Shape prior: the mean lung mask of the TRAIN split (a very simple 'atlas').
Used (a) as a baseline any real method must beat, (b) to restrict candidates to plausible lung
territory, (c) as a fallback when a method finds nothing. Built from train masks only (no leakage)."""
from pathlib import Path
import cv2
import numpy as np
from skimage.measure import label, regionprops

ATLAS_PATH = Path(__file__).resolve().parents[2] / "models" / "lung_atlas.npy"


def build_atlas(masks):
    """masks: iterable of binary (H,W) arrays -> float32 mean in [0,1]."""
    acc, n = None, 0
    for m in masks:
        m = (np.asarray(m) > 0).astype(np.float32)
        acc = m if acc is None else acc + m
        n += 1
    return acc / max(n, 1)


def save_atlas(atlas, path=ATLAS_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, atlas.astype(np.float32))


def load_atlas(path=ATLAS_PATH):
    return np.load(path) if Path(path).exists() else None


def atlas_mask(atlas, thresh=0.5):
    return (atlas > thresh).astype(np.uint8)


def allowed_region(atlas, min_freq=0.15, margin=21):
    """Pixels where a lung occurred in at least min_freq of training images, dilated by margin."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    return cv2.dilate((atlas > min_freq).astype(np.uint8), k).astype(bool)


def atlas_centers(atlas, thresh=0.5):
    """Centroids (row, col as fractions of H, W) of the two biggest atlas lobes, left then right."""
    lab = label(atlas > thresh, connectivity=2)
    props = sorted(regionprops(lab), key=lambda p: p.area, reverse=True)[:2]
    h, w = atlas.shape
    c = sorted([(p.centroid[0] / h, p.centroid[1] / w) for p in props], key=lambda t: t[1])
    return tuple(c) if len(c) == 2 else ((0.5, 0.28), (0.5, 0.72))
