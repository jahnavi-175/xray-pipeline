"""Unit VI classical lung segmentation baselines: thresholding, clustering, region growing.
Every method returns a binary uint8 mask (1 = lung) via segment_lungs(image, method)."""
import cv2
import numpy as np
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.measure import label
from skimage.segmentation import flood
from scipy.ndimage import binary_fill_holes
from .postprocess import select_lung_components, _ellipse
from .atlas import allowed_region, atlas_centers, atlas_mask

METHODS = ["otsu", "multi_otsu", "adaptive", "kmeans", "region_growing"]


def _smooth(img, sigma=3.0):
    """Gaussian smoothing; sigma ~3 px at 256x256 blurs rib stripes (~20 px apart) away."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def background_mask(s, thr=12):
    """Black image margin = very dark pixels connected to the image border. Thresholds and
    clusters must be computed on the BODY only, otherwise the margin hijacks the threshold."""
    lab = label(s < thr, connectivity=2)
    ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(ids))
    return bg if bg.mean() < 0.6 else np.zeros_like(bg)      # degenerate -> ignore


# ---- candidate generators: return a boolean "dark region" map (before clean-up) ----
def otsu_candidate(img):
    """2-class Otsu on body pixels: lung/air (dark) vs everything else."""
    s = _smooth(img); bg = background_mask(s)
    t = threshold_otsu(s[~bg])
    return (s < t) & ~bg


def multi_otsu_candidate(img):
    """3-class Otsu on body pixels: [lung/air] [soft tissue] [bone/metal]; keep the darkest class."""
    s = _smooth(img); bg = background_mask(s)
    try:
        t = threshold_multiotsu(s[~bg], classes=3)
    except ValueError:                       # almost-constant image
        return np.zeros(img.shape, bool)
    return (s < t[0]) & ~bg


def adaptive_candidate(img, block=91, c=0):
    """Pixel darker than its local (Gaussian-weighted) neighbourhood mean."""
    s = _smooth(img); bg = background_mask(s)
    m = cv2.adaptiveThreshold(s, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c)
    return m.astype(bool) & ~bg


def kmeans_candidate(img, k=3, min_area_frac=0.03, seed=0):
    """Cluster body-pixel intensities into k groups; the darkest is lung. If it is too
    fragmentary to look like lungs, merge in the next-darkest cluster."""
    s = _smooth(img); bg = background_mask(s)
    data = s[~bg].reshape(-1, 1).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lab1d, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = np.full(img.shape, -1, np.int32)
    labels[~bg] = lab1d.ravel()
    order = np.argsort(centers.ravel())
    cand = labels == order[0]
    opened = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(5))
    if select_lung_components(opened).sum() < min_area_frac * img.size:
        cand = cand | (labels == order[1])
    return cand


def auto_seeds(img, centers=((0.50, 0.33), (0.50, 0.67)), win=0.12):
    """One seed per lung: darkest smoothed non-black pixel near the expected lung centres
    (given as fractions of height/width)."""
    h, w = img.shape
    s = cv2.GaussianBlur(img, (0, 0), 4).astype(np.float32)
    s[s < 10] = 255                           # never seed inside the black margin
    seeds = []
    for fr, fc in centers:
        r0, c0 = int(fr * h), int(fc * w)
        dr, dc = int(win * h), int(win * w)
        win_ = s[max(r0 - dr, 0):r0 + dr, max(c0 - dc, 0):c0 + dc]
        rr, cc = np.unravel_index(np.argmin(win_), win_.shape)
        seeds.append((max(r0 - dr, 0) + rr, max(c0 - dc, 0) + cc))
    return seeds


def region_growing_candidate(img, tol=25, seeds=None, sigma=3.0, centers=None):
    """Grow from a seed in each lung: connected pixels within +-tol grey levels of the seed value.
    Heavy smoothing (sigma) suppresses rib stripes so the region does not stop at them; the seed
    value is the mean of a small patch rather than one (possibly noisy) pixel."""
    s = cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float64)
    seeds = seeds or (auto_seeds(img, centers) if centers else auto_seeds(img))
    mask = np.zeros(img.shape, bool)
    for r, c in seeds:
        patch = s[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4].mean()
        mask |= flood(s, (r, c), tolerance=tol) & (np.abs(s - patch) <= tol)
    return mask


CANDIDATES = {"otsu": otsu_candidate, "multi_otsu": multi_otsu_candidate,
              "adaptive": adaptive_candidate, "kmeans": kmeans_candidate,
              "region_growing": region_growing_candidate}


def segment_lungs(img, method="otsu", prior=None, fallback=True, **kw):
    """Main entry point (also used by the dashboard): uint8 grayscale image -> uint8 {0,1} mask.
    prior: optional mean-lung-mask atlas (see atlas.py). With a prior, candidates are limited to
    plausible lung territory and the atlas shape is returned if nothing lung-like is found."""
    if method not in CANDIDATES:
        raise ValueError(f"unknown method {method}; choose from {METHODS}")
    if prior is not None and method == "region_growing":
        kw.setdefault("centers", atlas_centers(prior))
    cand = CANDIDATES[method](img, **kw)
    if prior is not None:
        cand = cand & allowed_region(prior)
    m = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(15))
    sel = select_lung_components(m)                    # select BEFORE hole filling (a margin ring
    sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, _ellipse(11))   # would otherwise swallow the lungs)
    sel = binary_fill_holes(sel).astype(np.uint8)
    if fallback and prior is not None and sel.sum() < 0.03 * sel.size:
        return atlas_mask(prior)
    return sel
