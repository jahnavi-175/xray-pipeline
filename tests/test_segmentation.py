import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from tests.phantom import make_phantom
from src.segmentation import segment_lungs, METHODS
from src.segmentation.postprocess import keep_largest, clean_mask, remove_background
from src.evaluation.metrics import dice


@pytest.mark.parametrize("variant", ["wide", "touch", "tight"])
@pytest.mark.parametrize("method", METHODS)
def test_methods_segment_phantom(method, variant):
    img, gt = make_phantom(1, variant=variant)
    assert dice(segment_lungs(img, method), gt) > 0.85


@pytest.mark.parametrize("method", ["otsu", "adaptive", "kmeans", "region_growing"])
def test_black_margin_does_not_hijack_threshold(method):
    img, gt = make_phantom(1, variant="centered")      # big black margin round the body
    assert dice(segment_lungs(img, method), gt) > 0.75


def test_tight_crop_lungs_touching_border_are_found():
    img, gt = make_phantom(3, variant="tight")
    pred = segment_lungs(img, "kmeans")
    assert pred[:, 0].any() and dice(pred, gt) > 0.85       # prediction reaches the image edge


def test_atlas_prior_does_not_hurt_and_fallback_works():
    from src.segmentation import build_atlas
    gts = [make_phantom(s, variant="tight")[1] for s in range(4)]
    atlas = build_atlas(gts)
    img, gt = make_phantom(9, variant="tight")
    assert dice(segment_lungs(img, "otsu", prior=atlas), gt) >= dice(segment_lungs(img, "otsu"), gt) - 0.05
    blank = np.full((256, 256), 100, np.uint8)
    assert segment_lungs(blank, "otsu", prior=atlas).sum() > 0        # falls back to the atlas shape


@pytest.mark.parametrize("method", METHODS)
def test_output_is_binary_mask_of_right_shape(method):
    img, _ = make_phantom(2)
    m = segment_lungs(img, method)
    assert m.shape == img.shape and set(np.unique(m)) <= {0, 1}


def test_keep_largest_keeps_two_biggest():
    m = np.zeros((100, 100), np.uint8)
    m[5:30, 5:30] = 1; m[5:30, 60:90] = 1; m[80:84, 80:84] = 1
    out = keep_largest(m, 2, 0.0)
    assert out.sum() == 25 * 25 + 25 * 30 and out[81, 81] == 0


def test_constant_image_does_not_crash():
    for method in METHODS:
        segment_lungs(np.full((64, 64), 100, np.uint8), method)
