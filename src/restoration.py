"""Unit IV: image restoration  (degradation model  g = h * f + n).
All filters work in the frequency domain with circular convolution, matching degrade()."""
import numpy as np
import cv2
from .noise_models import add_gaussian, estimate_sigma


def gaussian_psf(sigma, ksize=None):
    ksize = ksize or int(2 * np.ceil(3 * sigma) + 1)
    ax = np.arange(ksize) - ksize // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def motion_psf(length=9, angle=0):
    k = np.zeros((length, length), np.float64)
    k[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, M, (length, length))
    return k / k.sum()


def psf2otf(psf, shape):
    """Zero-pad the PSF to `shape` and centre it at the origin, then FFT."""
    pad = np.zeros(shape, np.float64)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def blur(img, psf):
    """Float blur h * f (circular)."""
    H = psf2otf(psf, img.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(img.astype(np.float64)) * H))


def degrade(img, psf, noise_sigma=5.0, seed=0):
    """Simulate acquisition: blur, add Gaussian noise, quantise to uint8."""
    g = np.clip(np.rint(blur(img, psf)), 0, 255).astype(np.uint8)
    return add_gaussian(g, noise_sigma, seed) if noise_sigma > 0 else g


def _apply(g, W):
    G = np.fft.fft2(g.astype(np.float64))
    return np.clip(np.real(np.fft.ifft2(G * W)), 0, 255)


def inverse_filter(g, psf, eps=1e-8):
    """Naive inverse: F = G / H.  Explodes where |H| ~ 0 (shown on purpose in the report)."""
    H = psf2otf(psf, g.shape)
    H = np.where(np.abs(H) < eps, eps, H)
    return _apply(g, 1.0 / H).astype(np.uint8)


def truncated_inverse(g, psf, thresh=0.1):
    """Pseudo-inverse: only invert frequencies where |H| >= thresh (others zeroed)."""
    H = psf2otf(psf, g.shape)
    W = np.where(np.abs(H) >= thresh, 1.0 / np.where(H == 0, 1, H), 0.0)
    return _apply(g, W).astype(np.uint8)


def wiener_filter(g, psf, K=0.01):
    """Wiener with constant noise-to-signal ratio K:  W = H* / (|H|^2 + K)."""
    H = psf2otf(psf, g.shape)
    return _apply(g, np.conj(H) / (np.abs(H) ** 2 + K)).astype(np.uint8)


def estimate_K(g, floor=1e-5):
    """Crude automatic NSR: sigma_n^2 / signal variance (noise from Immerkaer estimate)."""
    s2 = estimate_sigma(g) ** 2
    sig = max(float(g.astype(np.float64).var()) - s2, 1.0)
    return max(s2 / sig, floor)


def wiener_auto(g, psf):
    return wiener_filter(g, psf, estimate_K(g))


def cls_filter(g, psf, gamma=0.01):
    """Constrained least squares (Laplacian smoothness):  W = H* / (|H|^2 + gamma |P|^2)."""
    H = psf2otf(psf, g.shape)
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float64)
    P = psf2otf(lap, g.shape)
    return _apply(g, np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)).astype(np.uint8)
