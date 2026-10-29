"""Unit VII texture: GLCM (inside the lung mask only) + first-order intensity statistics."""
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix
from .shape import two_lungs

ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def _bbox(mask):
    r, c = np.where(mask)
    return r.min(), r.max() + 1, c.min(), c.max() + 1


def glcm_features(img, mask, levels=32, distances=(1, 3)):
    """GLCM on the lung bounding box. Pixels outside the mask get level 0, and row/column 0 of the
    matrix is discarded, so only lung-to-lung pixel pairs count. Props are averaged over 4 angles."""
    r0, r1, c0, c1 = _bbox(mask)
    crop, m = img[r0:r1, c0:c1].astype(np.float64), mask[r0:r1, c0:c1]
    lo, hi = np.percentile(crop[m], (1, 99))
    q = np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1) * (levels - 2) + 1          # lung -> 1..levels-1
    q = np.where(m, np.floor(q), 0).astype(np.uint8)
    g = graycomatrix(q, list(distances), ANGLES, levels=levels, symmetric=True, normed=False)
    idx = np.arange(1, levels)
    i, j = idx[:, None].astype(float), idx[None, :].astype(float)
    diff = np.abs(i - j)
    out = {}
    for di, d in enumerate(distances):
        rows = []
        for ai in range(len(ANGLES)):
            P = g[1:, 1:, di, ai].astype(np.float64)
            s = P.sum()
            if s <= 0:
                continue
            P /= s
            nz = P[P > 0]
            mu_i, mu_j = (P * i).sum(), (P * j).sum()
            si = np.sqrt((P * (i - mu_i) ** 2).sum()); sj = np.sqrt((P * (j - mu_j) ** 2).sum())
            corr = (P * (i - mu_i) * (j - mu_j)).sum() / (si * sj) if si * sj > 1e-12 else 1.0
            asm = (P ** 2).sum()
            rows.append([(P * diff ** 2).sum(), (P * diff).sum(), (P / (1 + diff ** 2)).sum(),
                         np.sqrt(asm), asm, -(nz * np.log2(nz)).sum(), corr])
        vals = np.mean(rows, axis=0) if rows else [np.nan] * 7
        for name, v in zip(["contrast", "dissimilarity", "homogeneity", "energy", "asm", "entropy", "correlation"], vals):
            out[f"glcm_d{d}_{name}"] = float(v)
    return out


def stat_features(img, mask):
    """First-order statistics of intensities inside the lungs (+ lung-vs-rest contrast, L/R asymmetry)."""
    v = img[mask].astype(np.float64) / 255.0
    hist = np.histogram(v, bins=32, range=(0, 1))[0] / len(v)
    nz = hist[hist > 0]
    rest = img[(~mask) & (img > 10)]
    rest_mean = rest.mean() / 255.0 if rest.size else np.nan
    f = {"stat_mean": v.mean(), "stat_std": v.std(), "stat_skew": float(skew(v)), "stat_kurt": float(kurtosis(v)),
         "stat_entropy": float(-(nz * np.log2(nz)).sum()), "stat_p10": np.percentile(v, 10),
         "stat_p50": np.percentile(v, 50), "stat_p90": np.percentile(v, 90),
         "stat_lung_to_rest": v.mean() / rest_mean if rest_mean and rest_mean > 0 else np.nan}
    lungs = two_lungs(mask)
    if len(lungs) == 2:
        m1, m2 = (img[tuple(p.coords.T)].mean() / 255.0 for p in lungs)
        f["stat_lr_mean_diff"] = abs(m1 - m2) / max((m1 + m2) / 2, 1e-6)
    else:
        f["stat_lr_mean_diff"] = np.nan
    return {k: float(x) for k, x in f.items()}
