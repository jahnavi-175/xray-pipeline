"""Segmentation quality of a mask source, so the ablation table can show Dice next to accuracy."""
import cv2
import numpy as np
import pandas as pd
from ..io_utils import ROOT, load_mask
from .metrics import dice, iou


def segmentation_scores(cfg, masks, split="val", n=300, seed=0):
    """Mean Dice / IoU of a predicted-mask folder against the ground-truth masks.
    'gt' -> (1, 1);  'full' (no segmentation) -> (nan, nan)."""
    if masks == "gt":
        return 1.0, 1.0
    if masks == "full":
        return float("nan"), float("nan")
    df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
    df = df[(df.split == split) & df.mask_path.notna()]
    df = df.sample(min(n, len(df)), random_state=seed)
    d, j = [], []
    for r in df.itertuples():
        gt = load_mask(r.mask_path, cfg["image_size"])
        pr = (cv2.imread(str(ROOT / cfg["paths"]["predicted_masks"] / masks / f"{r.image_id}.png"),
                         cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        d.append(dice(pr, gt)); j.append(iou(pr, gt))
    return float(np.mean(d)), float(np.mean(j))
