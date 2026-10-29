"""Morphological clean-up shared by all classical segmenters (Unit VI / morphology)."""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def keep_largest(mask, n=2, min_area_frac=0.01):
    """Keep the n largest connected components (the two lungs), dropping specks."""
    lab = label(mask.astype(bool), connectivity=2)
    if lab.max() == 0:
        return np.zeros(mask.shape, np.uint8)
    areas = np.bincount(lab.ravel())[1:]
    order = np.argsort(areas)[::-1][:n]
    keep = [i + 1 for i in order if areas[i] >= min_area_frac * mask.size]
    return np.isin(lab, keep).astype(np.uint8)


def remove_background(candidate, min_frac=0.03, strict=False):
    """Dark candidates include the black image border. Lungs are enclosed by the body, so:
    1) drop components that touch the image border;
    2) if that removes almost everything (lungs touch the edge), only drop components containing
       an image corner; 3) if still empty, return the candidate unchanged
       (or an empty mask when strict=True, used to reject a candidate that is only background)."""
    c = candidate.astype(bool)
    out = clear_border(c)
    if out.sum() >= min_frac * c.size:
        return out.astype(np.uint8)
    lab = label(c, connectivity=2)
    h, w = c.shape
    corner_ids = {lab[0, 0], lab[0, w - 1], lab[h - 1, 0], lab[h - 1, w - 1]} - {0}
    out = c & ~np.isin(lab, list(corner_ids))
    if out.sum() >= min_frac * c.size:
        return out.astype(np.uint8)
    return np.zeros(c.shape, np.uint8) if strict else c.astype(np.uint8)


def clean_mask(mask, open_k=5, close_k=11, n=2, min_area_frac=0.01):
    """open (remove specks) -> close (bridge rib gaps) -> fill holes -> keep n largest."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(open_k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(close_k))
    m = binary_fill_holes(m).astype(np.uint8)
    return keep_largest(m, n, min_area_frac)


def select_lung_components(mask, row_range=(0.28, 0.72), min_area_frac=0.01, max_area_frac=0.6):
    """Pick the lungs from a set of dark regions using anatomy, NOT the image border
    (in tightly cropped X-rays the real lungs touch the border):
      * regions containing 2+ image corners or covering >60% of the image are background;
      * the centroid must lie in the middle band of the image (rejects dark air in the top corners
        and the stomach gas bubble / abdomen below the lungs);
      * choose at most one component per side (left/right), scored by area x closeness to the
        expected lung centre;
      * a single component spanning both sides (lungs bridged at the mediastinum) is kept alone."""
    m = mask.astype(bool)
    h, w = m.shape
    lab = label(m, connectivity=2)
    corners = [lab[0, 0], lab[0, w - 1], lab[h - 1, 0], lab[h - 1, w - 1]]
    best = {0: (0.0, None), 1: (0.0, None)}
    for p in regionprops(lab):
        if p.area < min_area_frac * h * w or p.area > max_area_frac * h * w:
            continue
        if corners.count(p.label) >= 2:          # black image margin / background ring
            continue
        cy, cx = p.centroid
        if not (row_range[0] * h <= cy <= row_range[1] * h):
            continue
        c0, c1 = p.bbox[1], p.bbox[3]
        if c0 < 0.35 * w and c1 > 0.65 * w and p.area > 0.12 * h * w:     # merged pair
            return (lab == p.label).astype(np.uint8)
        side = 0 if cx < w / 2 else 1
        ex_c = 0.28 if side == 0 else 0.72
        dist = np.hypot((cy / h - 0.5), (cx / w - ex_c))
        score = p.area * np.exp(-(dist / 0.25) ** 2)
        if score > best[side][0]:
            best[side] = (score, p.label)
    keep = [lbl for _, lbl in best.values() if lbl is not None]
    return np.isin(lab, keep).astype(np.uint8)
