import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.evaluation.metrics import dice, iou
from src.enhancement import clahe, homomorphic, butterworth_highpass, hist_specification
from src.noise_models import add_gaussian, estimate_sigma


def test_dice_iou_identical():
    m = np.zeros((10, 10)); m[2:6, 2:6] = 1
    assert abs(dice(m, m) - 1) < 1e-6 and abs(iou(m, m) - 1) < 1e-6


def test_dice_iou_disjoint():
    a = np.zeros((10, 10)); a[:3] = 1
    b = np.zeros((10, 10)); b[7:] = 1
    assert dice(a, b) < 1e-6 and iou(a, b) < 1e-6


def test_enhancements_keep_shape_dtype():
    img = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    for f in (clahe, homomorphic, butterworth_highpass):
        out = f(img)
        assert out.shape == img.shape and out.dtype == np.uint8
    assert hist_specification(img, img).shape == img.shape


def test_noise_estimate_tracks_sigma():
    flat = np.full((128, 128), 128, np.uint8)
    est = estimate_sigma(add_gaussian(flat, 10))
    assert 7 < est < 13
