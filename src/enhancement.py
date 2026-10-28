"""Unit II (spatial) and Unit III (frequency-domain) enhancement + quality measures."""
import cv2
import numpy as np
from skimage.exposure import match_histograms
from .preprocessing import to_uint8


# ---------- Spatial domain (Unit II) ----------
def hist_equalize(img):
    return cv2.equalizeHist(img)


def clahe(img, clip_limit=2.0, tile_grid=8):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)).apply(img)


def hist_specification(img, reference):
    """Match img's histogram to a reference image (e.g. a well-exposed 'typical' scan)."""
    return match_histograms(img, reference).astype(np.uint8)


def gamma(img, g=0.8):
    return (255 * (img / 255.0) ** g).astype(np.uint8)


# ---------- Frequency domain (Unit III) ----------
def _dist_grid(shape):
    h, w = shape
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U ** 2 + V ** 2)


def homomorphic_H(shape, gamma_low=0.5, gamma_high=1.8, c=1.0, d0=30):
    """Transfer function of the homomorphic filter (exposed so it can be plotted)."""
    D = _dist_grid(shape)
    return (gamma_high - gamma_low) * (1 - np.exp(-c * D ** 2 / (d0 ** 2))) + gamma_low


def homomorphic(img, gamma_low=0.5, gamma_high=1.8, c=1.0, d0=30):
    """log -> FFT -> Gaussian high-emphasis -> IFFT -> exp.
    Suppresses slow illumination variation (low freq) and boosts detail/reflectance (high freq)."""
    f = np.log1p(img.astype(np.float64))
    F = np.fft.fftshift(np.fft.fft2(f))
    H = homomorphic_H(img.shape, gamma_low, gamma_high, c, d0)
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return to_uint8(np.expm1(out))


def butterworth_highpass(img, d0=30, order=2, boost=1.0):
    """High-frequency emphasis with a Butterworth high-pass: H = 1 + boost * HP."""
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    D = _dist_grid(img.shape)
    D[D == 0] = 1e-6
    HP = 1.0 / (1.0 + (d0 / D) ** (2 * order))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * (1 + boost * HP))))
    return to_uint8(out)


def enhance(img, method, cfg=None, reference=None):
    """Single entry point used by scripts and the dashboard."""
    cfg = cfg or {}
    if method == "none":
        return img
    if method == "he":
        return hist_equalize(img)
    if method == "clahe":
        return clahe(img, **cfg.get("clahe", {}))
    if method == "histspec":
        if reference is None:
            raise ValueError("histspec needs a reference image")
        return hist_specification(img, reference)
    if method == "homomorphic":
        return homomorphic(img, **cfg.get("homomorphic", {}))
    if method == "highpass":
        return butterworth_highpass(img, **cfg.get("highpass", {}))
    raise ValueError(method)


METHODS = ["none", "he", "clahe", "histspec", "homomorphic", "highpass"]


# ---------- Quality measures (no-reference) ----------
def entropy(img):
    p = np.bincount(img.ravel(), minlength=256) / img.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rms_contrast(img):
    return float(img.astype(np.float64).std())


def mean_gradient(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    return float(np.mean(np.hypot(gx, gy)))


def quality_report(img):
    return dict(entropy=entropy(img), contrast=rms_contrast(img), sharpness=mean_gradient(img))
