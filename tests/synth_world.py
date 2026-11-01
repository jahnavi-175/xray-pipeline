"""A tiny synthetic 'trained system' (classifier bundle + atlas + images) for pipeline / app tests."""
import pandas as pd
from tests.phantom import make_phantom
from src.features import extract_features
from src.segmentation import build_atlas
from src.classifiers.classical_ml import build_pipeline


def build_world(n=60):
    """Class 1 = noisier phantoms. Returns (bundle, atlas, images, ground_truth_masks)."""
    rows, gts, imgs = [], [], []
    for i in range(n):
        y = i % 2
        img, gt = make_phantom(i, variant="tight", noise=3 + 12 * y)
        rows.append({"label": y, **extract_features(img, gt)}); gts.append(gt); imgs.append(img)
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.split("_")[0] in ("shape", "stat", "glcm", "hog") and df[c].nunique() > 1]
    pipe = build_pipeline("rf").set_params(clf__n_estimators=50).fit(df[cols], df.label)
    bundle = {"pipeline": pipe, "columns": cols, "enh": "none", "kind": "rf", "masks": "none_unet",
              "groups": ["shape", "stat", "glcm", "hog"],
              "val_scores": pipe.predict_proba(df[cols])[:, 1], "val_labels": df.label.values,
              "reference": {"shape_area_frac": {"normal": 0.5, "abnormal": 0.4},
                            "stat_mean": {"normal": 0.3, "abnormal": 0.5}}}
    return bundle, build_atlas(gts), imgs, gts
