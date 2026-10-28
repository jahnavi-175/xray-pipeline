"""Unit IV: noise models (to simulate) and noise estimation."""
import numpy as np
import cv2


def _rng(seed):
    return np.random.default_rng(seed)


def add_gaussian(img, sigma=15, seed=0):
    out = img.astype(np.float32) + _rng(seed).normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.02, seed=0):
    out = img.copy()
    r = _rng(seed).random(img.shape)
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out


def add_speckle(img, sigma=0.1, seed=0):
    f = img.astype(np.float32)
    out = f + f * _rng(seed).normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_poisson(img, seed=0):
    """Quantum (photon) noise: the dominant physical noise in X-ray imaging."""
    scale = 255.0 / 40  # lower -> noisier
    out = _rng(seed).poisson(img.astype(np.float32) / 255.0 * 40) * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def blur_and_noise(img, ksize=7, sigma_blur=2.0, sigma_noise=5, seed=0):
    """Degradation model g = h*f + n, used for the restoration experiments (Part 2)."""
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma_blur)
    return add_gaussian(blurred, sigma_noise, seed)


def estimate_sigma(img):
    """Immerkaer fast noise-sigma estimate (Laplacian-difference mask)."""
    f = img.astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = f.shape
    conv = cv2.filter2D(f, -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (h - 2) * (w - 2)))
