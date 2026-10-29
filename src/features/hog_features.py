"""Unit VII: Histogram of Oriented Gradients on the lung region."""
import cv2
import numpy as np
from skimage.feature import hog


def hog_features(img, mask, size=128, orientations=9, ppc=32, cpb=2, margin=0.05):
    """Crop to the lung bounding box (+margin), paint everything outside the lungs with the mean lung
    value (so the mask edge does not dominate the gradients), resize and run HOG.
    Default: 128 px, 32 px cells, 2x2 blocks, 9 bins -> 324 numbers."""
    r, c = np.where(mask)
    h, w = img.shape
    dr, dc = int(margin * h), int(margin * w)
    r0, r1, c0, c1 = max(r.min() - dr, 0), min(r.max() + 1 + dr, h), max(c.min() - dc, 0), min(c.max() + 1 + dc, w)
    crop = img[r0:r1, c0:c1].astype(np.float32)
    m = mask[r0:r1, c0:c1]
    crop = np.where(m, crop, crop[m].mean())
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    v = hog(crop, orientations=orientations, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb),
            block_norm="L2-Hys", feature_vector=True)
    return {f"hog_{i:03d}": float(x) for i, x in enumerate(v)}
