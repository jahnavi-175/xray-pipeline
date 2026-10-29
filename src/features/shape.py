"""Unit VII shape descriptors computed from the (predicted) lung mask."""
import cv2
import numpy as np
from skimage.measure import label, regionprops


def two_lungs(mask, min_frac=0.005):
    """The (up to) two biggest components, ordered left-to-right IN THE IMAGE."""
    lab = label(mask > 0, connectivity=2)
    props = [p for p in regionprops(lab) if p.area >= min_frac * mask.size]
    props = sorted(props, key=lambda p: p.area, reverse=True)[:2]
    return sorted(props, key=lambda p: p.centroid[1])


def _side_names(lungs, w):
    """Assign 'L'/'R' (image-left / image-right) even if only one lung component was found."""
    if len(lungs) == 2:
        return {"L": lungs[0], "R": lungs[1]}
    if len(lungs) == 1:
        return {("L" if lungs[0].centroid[1] < w / 2 else "R"): lungs[0]}
    return {}


def shape_features(mask):
    h, w = mask.shape
    lungs = two_lungs(mask)
    sides = _side_names(lungs, w)
    union = np.zeros(mask.shape, bool)
    for p in lungs:
        union[tuple(p.coords.T)] = True
    f = {"shape_area_frac": union.mean(), "shape_n_lungs": float(len(lungs))}

    cnts, _ = cv2.findContours(union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    f["shape_perimeter_norm"] = per / (2.0 * (h + w))
    hull_area = cv2.contourArea(cv2.convexHull(np.concatenate(cnts))) if cnts else 0.0
    f["shape_solidity"] = min(union.sum() / hull_area, 1.0) if hull_area > 0 else np.nan

    rows, cols = np.where(union)
    bh, bw = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    f["shape_bbox_aspect"] = bh / bw
    f["shape_bbox_fill"] = union.sum() / (bh * bw)
    hu = cv2.HuMoments(cv2.moments(union.astype(np.uint8), binaryImage=True)).ravel()[:4]
    for i, v in enumerate(hu):
        f[f"shape_hu{i + 1}"] = np.sign(v) * np.log10(abs(v) + 1e-12)

    for s in ("L", "R"):
        p = sides.get(s)
        f[f"shape_{s}_area_frac"] = p.area / mask.size if p else np.nan
        f[f"shape_{s}_solidity"] = p.solidity if p else np.nan
        f[f"shape_{s}_eccentricity"] = p.eccentricity if p else np.nan
        f[f"shape_{s}_extent"] = p.extent if p else np.nan
        f[f"shape_{s}_cx"] = p.centroid[1] / w if p else np.nan
        f[f"shape_{s}_cy"] = p.centroid[0] / h if p else np.nan
        f[f"shape_{s}_orientation"] = p.orientation if p else np.nan
    if "L" in sides and "R" in sides:
        a, b = sides["L"], sides["R"]
        f["shape_lr_area_ratio"] = min(a.area, b.area) / max(a.area, b.area)
        f["shape_lr_cy_diff"] = abs(a.centroid[0] - b.centroid[0]) / h
        gap = max(b.bbox[1] - a.bbox[3], 0)                       # space between the lungs (mediastinum)
        f["shape_medial_gap_ratio"] = gap / max(b.bbox[3] - a.bbox[1], 1)
    else:
        f["shape_lr_area_ratio"] = f["shape_lr_cy_diff"] = f["shape_medial_gap_ratio"] = np.nan
    return {k: float(v) for k, v in f.items()}
