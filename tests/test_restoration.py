import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from skimage import data
import cv2
from src.restoration import gaussian_psf, blur, degrade, inverse_filter, wiener_filter, wiener_auto
from src.evaluation.metrics import psnr

F = cv2.resize(data.camera(), (256, 256), interpolation=cv2.INTER_AREA)


def test_psf_sums_to_one():
    assert abs(gaussian_psf(2).sum() - 1) < 1e-9


def test_inverse_perfect_when_noise_free_float():
    psf = gaussian_psf(1.0)
    g = blur(F, psf)                                    # float, no quantisation
    from src.restoration import psf2otf
    rec = np.real(np.fft.ifft2(np.fft.fft2(g) / psf2otf(psf, F.shape)))
    assert np.abs(rec - F).max() < 1e-3


def test_wiener_beats_degraded_and_inverse_under_noise():
    psf = gaussian_psf(2.0)
    g = degrade(F, psf, 5.0, seed=0)
    assert psnr(F, wiener_filter(g, psf, 1e-2)) > psnr(F, g)
    assert psnr(F, wiener_filter(g, psf, 1e-2)) > psnr(F, inverse_filter(g, psf)) + 5
    assert psnr(F, wiener_auto(g, psf)) > psnr(F, inverse_filter(g, psf))
