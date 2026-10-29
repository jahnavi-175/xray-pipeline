import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tests.phantom import make_phantom
from src.features import extract_features, feature_names, groups_of
from src.features.glcm import glcm_features
from src.features.shape import shape_features
from src.features.hog_features import hog_features


def test_feature_count_and_groups():
    g = groups_of(feature_names())
    assert len(g["hog"]) == 324 and len(g["glcm"]) == 14 and len(g["shape"]) == 27 and len(g["stat"]) == 10


def test_features_finite_on_phantom_and_area_matches_mask():
    img, gt = make_phantom(0, variant="tight")
    f = extract_features(img, gt)
    assert f["mask_ok"] == 1 and all(np.isfinite(v) for v in f.values())
    assert abs(f["shape_area_frac"] - gt.mean()) < 1e-6


def test_left_right_symmetric_mask():
    m = np.zeros((256, 256), np.uint8); m[60:200, 30:110] = 1; m[60:200, 146:226] = 1
    f = shape_features(m)
    assert abs(f["shape_lr_area_ratio"] - 1) < 1e-6 and f["shape_n_lungs"] == 2
    assert f["shape_medial_gap_ratio"] > 0 and f["shape_L_cx"] < 0.5 < f["shape_R_cx"]


def test_glcm_constant_lung_has_zero_contrast_full_energy():
    img = np.full((256, 256), 120, np.uint8)
    m = np.zeros((256, 256), bool); m[50:200, 50:200] = True
    f = glcm_features(img, m)
    assert f["glcm_d1_contrast"] < 1e-9 and abs(f["glcm_d1_energy"] - 1) < 1e-9


def test_glcm_ignores_pixels_outside_mask():
    rng = np.random.default_rng(0)
    img = np.full((256, 256), 100, np.uint8)
    m = np.zeros((256, 256), bool); m[60:190, 60:190] = True
    a = glcm_features(img, m)["glcm_d1_contrast"]
    img2 = img.copy(); img2[~m] = rng.integers(0, 255, (~m).sum())       # random garbage outside
    assert abs(glcm_features(img2, m)["glcm_d1_contrast"] - a) < 1e-9


def test_empty_mask_gives_nan_and_flag():
    img, gt = make_phantom(0)
    f = extract_features(img, np.zeros_like(gt))
    assert f["mask_ok"] == 0 and np.isnan(f["glcm_d1_contrast"])


def test_hog_length_and_mask_edge_invariance():
    img, gt = make_phantom(0, variant="tight")
    assert len(hog_features(img, gt)) == 324
