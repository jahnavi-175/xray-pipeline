"""Resize, normalise, denoise."""
import cv2
import numpy as np


def resize(img, size=256, interp=cv2.INTER_AREA):
    return cv2.resize(img, (size, size), interpolation=interp)


def normalize01(img):
    """uint8 -> float32 in [0,1]."""
    return img.astype(np.float32) / 255.0


def to_uint8(x, clip=None):
    """Float image -> uint8 by min-max stretch.
    clip=(lo, hi) percentiles: ignore extreme outliers (bright text markers, metal) so they
    do not squash the rest of the image into the dark end of the range."""
    x = x.astype(np.float64)
    if clip is not None:
        lo, hi = np.percentile(x, clip)
        x = np.clip(x, lo, hi)
    rng = x.max() - x.min()
    if rng < 1e-12:
        return np.zeros(x.shape, np.uint8)
    return ((x - x.min()) / rng * 255).astype(np.uint8)


def denoise(img, method="median", ksize=3):
    if method == "none":
        return img
    if method == "median":
        return cv2.medianBlur(img, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 7, 50, 50)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(img, None, h=10)
    raise ValueError(method)


def preprocess(img, size=256, denoise_method="median", ksize=3):
    """Standard Part-1 pipeline. Returns uint8 (store) ; call normalize01 before ML."""
    return denoise(resize(img, size), denoise_method, ksize)
