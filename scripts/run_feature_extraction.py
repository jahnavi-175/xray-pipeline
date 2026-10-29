"""Part 4: build the per-image feature table.

  python scripts/run_feature_extraction.py --enh none --masks none_unet       # real pipeline
  python scripts/run_feature_extraction.py --enh none --masks gt              # oracle: ground-truth masks
  python scripts/run_feature_extraction.py --enh none --masks full            # ablation: no segmentation
--masks : 'gt' | 'full' | name of a folder in data/predicted_masks (e.g. none_unet)
Output: data/features/features_<enh>__<masks>.csv  (id, label, split, mask_ok, shape_*, stat_*, glcm_*, hog_*)"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from src.io_utils import ROOT, load_config, load_gray, load_mask, enhanced_path
from src.features import extract_features


def one(row, cfg, enh, masks):
    img = load_gray(enhanced_path(cfg, enh, row["image_id"]))
    size = cfg["image_size"]
    if masks == "gt":
        m = load_mask(row["mask_path"], size)
    elif masks == "full":
        m = np.ones(img.shape, np.uint8)
    else:
        p = ROOT / cfg["paths"]["predicted_masks"] / masks / f"{row['image_id']}.png"
        m = (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    f = extract_features(img, m)
    return {"image_id": row["image_id"], "label": row["label"], "split": row["split"],
            "class_name": row["class_name"], **f}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enh", default="none")
    ap.add_argument("--masks", default="none_unet")
    ap.add_argument("--jobs", type=int, default=-1, help="-1 = all CPU cores")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick test)")
    args = ap.parse_args()

    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
    if args.masks == "gt":
        df = df[df.mask_path.notna()]
    if args.limit:
        df = df.sample(args.limit, random_state=cfg["seed"])
    rows = df.to_dict("records")
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, batch_size=16)(delayed(one)(r, cfg, args.enh, args.masks) for r in tqdm(rows))
    tab = pd.DataFrame(res)
    out = ROOT / cfg["paths"]["features"] / f"features_{args.enh}__{args.masks}.csv"
    tab.to_csv(out, index=False, float_format="%.6g")
    feat_cols = [c for c in tab.columns if c.split("_")[0] in ("shape", "stat", "glcm", "hog")]
    print(f"\n{len(tab)} images x {len(feat_cols)} features in {(time.time() - t0) / 60:.1f} min -> {out}")
    print("bad masks (mask_ok=0):", int((tab.mask_ok == 0).sum()))
    nan_cols = tab[feat_cols].isna().mean().sort_values(ascending=False)
    print("features with >5% NaN:", nan_cols[nan_cols > 0.05].round(3).to_dict() or "none")
