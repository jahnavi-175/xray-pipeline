"""Data helpers for the U-Net (NumPy / OpenCV only, no PyTorch needed)."""
import cv2
import numpy as np
from tqdm import tqdm
from ..io_utils import load_gray, load_mask, enhanced_path
from ..evaluation.compare import sample_rows


def resize_pair(img, mask, size):
    """Image with INTER_AREA; mask via area-averaging then 0.5 threshold (keeps it binary)."""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    y = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return x, (y > 0.5).astype(np.uint8)


def load_arrays(df, cfg, enh, size, n_total=None, seed=0):
    """Preload images + masks into memory. n_total (optional) = class-balanced subsample size."""
    df = df[df.mask_path.notna()]
    if n_total:
        df = sample_rows(df, max(1, n_total // df.class_name.nunique()), seed)
    X, Y = [], []
    for r in tqdm(list(df.itertuples()), desc=f"loading ({enh}, {size}px)"):
        img = load_gray(enhanced_path(cfg, enh, r.image_id))
        x, y = resize_pair(img, load_mask(r.mask_path, cfg["image_size"]), size)
        X.append(x); Y.append(y)
    return np.stack(X), np.stack(Y)


def augment(img, mask, rng, max_rot=10, scale=(0.9, 1.1), shift=0.05):
    """Small random rotation/scale/shift + brightness/contrast jitter. No flips (heart side matters)."""
    h, w = img.shape
    ang = rng.uniform(-max_rot, max_rot)
    sc = rng.uniform(*scale)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += rng.uniform(-shift, shift, 2) * (w, h)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a, b = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img, mask
