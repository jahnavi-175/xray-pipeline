"""The whole analysis chain in one place (used by the Streamlit dashboard, the demo script and the tests):
upload -> preprocess (+restore if blurry) -> enhance -> segment -> features -> prediction.
Mirrors exactly what scripts/run_preprocessing.py + run_feature_extraction.py did for the training data."""
import os
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from .io_utils import ROOT, load_config
from .preprocessing import preprocess, denoise
from .enhancement import enhance, METHODS
from .restoration import audit_image, gaussian_psf, wiener_filter
from .segmentation import segment_lungs, load_atlas
from .features import extract_features

CLASSICAL = ["adaptive", "otsu", "kmeans", "multi_otsu", "region_growing"]


class Models:
    """Everything that is loaded from disk once. Any part may be missing; analyze() degrades gracefully."""

    def __init__(self, unet=None, clf=None, atlas=None, blur_thr=None, hist_ref=None, notes=None, cnn=None):
        self.unet, self.clf, self.atlas, self.cnn = unet, clf, atlas, cnn
        self.blur_thr, self.hist_ref, self.notes = blur_thr, hist_ref, notes or []


def load_models(models_dir=None, cfg=None):
    cfg = cfg or load_config()
    md = Path(models_dir) if models_dir else ROOT / "models"
    m = Models()
    unet_path = md / "unet_none.pt"
    if unet_path.exists():
        try:
            from .segmentation.unet import UNetSegmenter           # needs PyTorch
            m.unet = UNetSegmenter.load(unet_path, "cpu")
        except Exception as e:                                       # noqa: BLE001
            m.notes.append(f"U-Net could not be loaded ({type(e).__name__}); using a classical segmenter instead.")
    else:
        m.notes.append("models/unet_none.pt not found; using a classical segmenter instead.")
    cnn_path = md / "cnn_none.pt"
    if cnn_path.exists():
        try:
            from .classifiers.cnn import CNNPredictor               # needs PyTorch + torchvision
            m.cnn = CNNPredictor.load(cnn_path, "cpu")
        except Exception as e:                                       # noqa: BLE001
            m.notes.append(f"CNN could not be loaded ({type(e).__name__}); showing the feature-based model only.")
    clf_path = md / "classifier.joblib"
    if clf_path.exists():
        m.clf = joblib.load(clf_path)
    else:
        m.notes.append("models/classifier.joblib not found; prediction is disabled.")
    atlas_path = md / "lung_atlas.npy"
    m.atlas = np.load(atlas_path) if atlas_path.exists() else load_atlas()
    ref = md / "histspec_reference.png"
    if ref.exists():
        m.hist_ref = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
    audit = ROOT / cfg["paths"]["reports"] / "restoration_audit.csv"
    if audit.exists():
        m.blur_thr = float(np.percentile(pd.read_csv(audit).blur_score, cfg["restoration"]["blur_percentile"]))
    return m


def to_gray_uint8(img):
    """Accept uint8/uint16/float, gray or colour -> uint8 grayscale."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3].astype(np.uint8) if img.dtype != np.uint8 else img[..., :3], cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img.astype(np.float64) - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
    return img


def overlay_mask(img, mask, color=(0, 220, 90), alpha=0.28):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    m = mask > 0
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, np.float32)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = rgb.astype(np.uint8)
    cv2.drawContours(out, cs, -1, color, 1)
    return out


def available_enhancements(models):
    return [m for m in METHODS if m != "histspec" or models.hist_ref is not None]


def segment_with(proc, models, name):
    """name: 'unet' or a classical method, optionally with '+prior'. Returns (mask, method_used, note)."""
    note = None
    if name == "unet" and models.unet is None:
        name = "adaptive+prior" if models.atlas is not None else "adaptive"
        note = f"U-Net unavailable, used '{name}' instead (much less accurate)."
    if name.endswith("+prior") and models.atlas is None:
        name, note = name[:-6], (note or "") + " Shape prior unavailable (models/lung_atlas.npy missing)."
    if name == "unet":
        return models.unet(proc), name, note
    return segment_lungs(proc, name.split("+")[0], prior=models.atlas if name.endswith("+prior") else None), name, note


def analyze(img, models, cfg=None, seg="unet", enh_view="none", restore=True, threshold=0.5):
    """Run the full chain on one image. Returns a dict with every intermediate result."""
    cfg = cfg or load_config()
    t = {}
    out = {"warnings": [], "seg_requested": seg, "threshold": threshold}
    size, pp, rc = cfg["image_size"], cfg["preprocessing"], cfg["restoration"]

    t0 = time.time()
    base = preprocess(to_gray_uint8(img), size, "none")                      # resized, not yet denoised
    out["original"] = base
    aud = audit_image(base)
    out.update(blur_score=aud["blur_score"], noise_sigma=aud["sigma_noise"])
    proc = denoise(base, pp["denoise"], pp["denoise_ksize"])
    flagged = models.blur_thr is not None and aud["blur_score"] < models.blur_thr
    out["flagged_blurry"], out["restored"] = bool(flagged), False
    if restore and flagged:
        proc = wiener_filter(proc, gaussian_psf(rc["psf_sigma"]), rc["K"])
        out["restored"] = True
    out["processed"] = proc
    t["preprocess"] = time.time() - t0

    t0 = time.time()
    out["enhanced_view"] = enhance(proc, enh_view, cfg["enhancement"], models.hist_ref)
    feat_enh = models.clf["enh"] if models.clf else "none"
    if feat_enh == "histspec" and models.hist_ref is None:
        out["warnings"].append("The classifier needs the histogram-specification reference image (models/histspec_reference.png).")
        feat_enh = "none"
    out["feature_enh"] = feat_enh
    feat_img = enhance(proc, feat_enh, cfg["enhancement"], models.hist_ref)
    t["enhance"] = time.time() - t0

    t0 = time.time()
    mask, used, note = segment_with(proc, models, seg)
    if note:
        out["warnings"].append(note)
    out["seg_used"], out["mask"] = used, mask
    out["overlay"] = overlay_mask(proc, mask)
    out["lung_area_pct"] = 100.0 * float((mask > 0).mean())
    t["segment"] = time.time() - t0

    t0 = time.time()
    feats = extract_features(feat_img, mask)
    out["features"] = feats
    t["features"] = time.time() - t0

    out["prob"] = out["pred"] = out["cnn_prob"] = None
    if models.cnn is not None and feats["mask_ok"] == 1:
        out["cnn_prob"] = float(models.cnn.predict_prob(enhance(proc, models.cnn.enh, cfg["enhancement"], models.hist_ref)))
    if feats["mask_ok"] == 0:
        out["warnings"].append("Segmentation found no plausible lungs, so no prediction is made. "
                               "Check that this is a frontal chest X-ray.")
    elif models.clf is None:
        out["warnings"].append("No trained classifier available.")
    else:
        t0 = time.time()
        cols = models.clf["columns"]
        X = pd.DataFrame([{c: feats.get(c, np.nan) for c in cols}])
        out["prob"] = float(models.clf["pipeline"].predict_proba(X)[0, 1])
        out["pred"] = int(out["prob"] >= threshold)
        t["predict"] = time.time() - t0
        if out["lung_area_pct"] < 8 or out["lung_area_pct"] > 60:
            out["warnings"].append(f"Unusual lung area ({out['lung_area_pct']:.0f}% of the image): check the overlay.")
    out["timings"] = t
    return out


def operating_points(bundle, thresholds=None):
    """Sensitivity / specificity at each threshold, from the validation scores stored with the model."""
    if bundle is None or "val_scores" not in bundle:
        return None
    s, y = np.asarray(bundle["val_scores"]), np.asarray(bundle["val_labels"])
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    rows = []
    for th in thresholds:
        p = s >= th
        rows.append({"threshold": float(th), "sensitivity": float(p[y == 1].mean()),
                     "specificity": float((~p)[y == 0].mean()), "accuracy": float((p == y).mean())})
    return pd.DataFrame(rows)


def threshold_for_sensitivity(bundle, target=0.90):
    """Highest threshold that still reaches the target sensitivity on the validation set."""
    ops = operating_points(bundle, np.linspace(0.01, 0.99, 99))
    if ops is None:
        return None
    ok = ops[ops.sensitivity >= target]
    return None if ok.empty else ok.iloc[-1].to_dict()


def reference_table(feats, bundle):
    """This image's value for the interpretable features next to typical Normal / Abnormal training medians."""
    if bundle is None or not bundle.get("reference"):
        return None
    rows = []
    for name, ref in bundle["reference"].items():
        v = feats.get(name, np.nan)
        closer = "-" if not np.isfinite(v) else ("Normal" if abs(v - ref["normal"]) <= abs(v - ref["abnormal"]) else "Abnormal")
        rows.append({"feature": name, "this image": v, "Normal median": ref["normal"],
                     "Abnormal median": ref["abnormal"], "closer to": closer})
    return pd.DataFrame(rows)


def sample_dir():
    return Path(os.environ.get("XRAY_SAMPLES_DIR", ROOT / "app" / "samples"))
