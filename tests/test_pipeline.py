import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from tests.synth_world import build_world
from src.pipeline import (Models, analyze, to_gray_uint8, operating_points, threshold_for_sensitivity,
                          reference_table, overlay_mask)


@pytest.fixture(scope="module")
def world():
    bundle, atlas, imgs, gts = build_world()
    return Models(unet=None, clf=bundle, atlas=atlas, blur_thr=None), imgs, gts


def test_analyze_full_chain_with_classical_fallback(world):
    models, imgs, gts = world
    r = analyze(imgs[1], models, seg="unet")
    assert r["seg_used"] == "adaptive+prior" and any("U-Net unavailable" in w for w in r["warnings"])
    assert r["mask"].shape == (256, 256) and r["mask"].sum() > 0
    assert 0.0 <= r["prob"] <= 1.0 and r["pred"] in (0, 1)
    assert len(r["features"]) == 376 and r["overlay"].shape == (256, 256, 3)
    assert set(r["timings"]) >= {"preprocess", "segment", "features", "predict"}


def test_unet_branch_and_threshold_changes_decision(world):
    models, imgs, gts = world
    m = Models(unet=lambda im: gts[0], clf=models.clf, atlas=models.atlas)
    lo, hi = analyze(imgs[0], m, seg="unet", threshold=0.0), analyze(imgs[0], m, seg="unet", threshold=1.01)
    assert lo["seg_used"] == "unet" and not lo["warnings"]
    assert lo["pred"] == 1 and hi["pred"] == 0 and lo["prob"] == hi["prob"]


def test_model_separates_the_synthetic_classes(world):
    models, imgs, gts = world
    p0 = np.mean([analyze(imgs[i], models, seg="otsu+prior")["prob"] for i in range(0, 20, 2)])
    p1 = np.mean([analyze(imgs[i], models, seg="otsu+prior")["prob"] for i in range(1, 21, 2)])
    assert p1 > p0 + 0.2


def test_blank_image_gives_no_prediction_and_a_warning(world):
    models, _, _ = world
    r = analyze(np.full((300, 300), 90, np.uint8), models, seg="adaptive")
    assert r["prob"] is None and any("no plausible lungs" in w for w in r["warnings"])


def test_no_classifier_degrades_gracefully(world):
    models, imgs, _ = world
    r = analyze(imgs[0], Models(atlas=models.atlas), seg="kmeans+prior")
    assert r["prob"] is None and any("No trained classifier" in w for w in r["warnings"])


def test_restoration_applies_only_when_flagged(world):
    models, imgs, _ = world
    m = Models(clf=models.clf, atlas=models.atlas, blur_thr=1e9)          # everything counts as blurry
    assert analyze(imgs[0], m, seg="otsu", restore=True)["restored"] is True
    assert analyze(imgs[0], m, seg="otsu", restore=False)["restored"] is False
    m2 = Models(clf=models.clf, atlas=models.atlas, blur_thr=-1.0)
    assert analyze(imgs[0], m2, seg="otsu")["restored"] is False


def test_input_conversion():
    rgb = np.stack([np.full((50, 60), 100, np.uint8)] * 3, -1)
    assert to_gray_uint8(rgb).shape == (50, 60) and to_gray_uint8(rgb).dtype == np.uint8
    g16 = (np.arange(100 * 100).reshape(100, 100) * 6).astype(np.uint16)
    o = to_gray_uint8(g16)
    assert o.dtype == np.uint8 and o.min() == 0 and o.max() == 255


def test_operating_points_and_target_sensitivity(world):
    models, _, _ = world
    ops = operating_points(models.clf)
    assert (np.diff(ops.sensitivity) <= 1e-12).all() and (np.diff(ops.specificity) >= -1e-12).all()
    best = threshold_for_sensitivity(models.clf, 0.9)
    assert best is None or best["sensitivity"] >= 0.9
    assert operating_points(None) is None


def test_reference_table_and_overlay(world):
    models, imgs, gts = world
    r = analyze(imgs[0], models, seg="otsu+prior")
    t = reference_table(r["features"], models.clf)
    assert list(t.columns)[:2] == ["feature", "this image"] and set(t["closer to"]) <= {"Normal", "Abnormal", "-"}
    ov = overlay_mask(imgs[0], gts[0])
    assert ov.shape == (256, 256, 3) and not np.array_equal(ov[..., 0], ov[..., 1])


def test_cnn_second_opinion_is_used_when_available_and_skipped_for_bad_masks(world):
    from types import SimpleNamespace
    models, imgs, gts = world
    cnn = SimpleNamespace(enh="none", predict_prob=lambda im: 0.77)
    m = Models(unet=lambda im: gts[0], clf=models.clf, atlas=models.atlas, cnn=cnn)
    assert analyze(imgs[0], m, seg="unet")["cnn_prob"] == 0.77
    assert analyze(imgs[0], models, seg="unet")["cnn_prob"] is None            # no CNN loaded
    blank = Models(unet=lambda im: np.zeros_like(im), clf=models.clf, atlas=models.atlas, cnn=cnn)
    assert analyze(imgs[0], blank, seg="unet")["cnn_prob"] is None            # empty mask -> no prediction at all
