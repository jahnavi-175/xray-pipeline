"""Resize, normalise, denoise."""
import cv2
import numpy as np


def resize(img, size=256, interp=cv2.INTER_AREA):
    return cv2.resize(img, (size, size), interpolation=interp)


def normalize01(img):
    """uint8 -> float32 in [0,1]."""
    return img.astype(np.float32) / 255.0


def to_uint8(x):
    """Any float image -> min-max stretched uint8."""
    x = x.astype(np.float64)
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
