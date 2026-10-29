"""Part 3: build the mean-lung-mask atlas from the TRAIN split and report how well the atlas alone
(no image information at all) scores on the validation split. Any real method must beat this.
Usage: python scripts/build_atlas.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io_utils import ROOT, load_config, load_mask
from src.segmentation.atlas import build_atlas, save_atlas, atlas_mask
from src.evaluation.metrics import dice, iou

cfg = load_config(); size = cfg["image_size"]
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
df = df[df.mask_path.notna()]
tr, va = df[df.split == "train"], df[df.split == "val"]

atlas = build_atlas(load_mask(r.mask_path, size) for r in tqdm(tr.itertuples(), total=len(tr), desc="train masks"))
save_atlas(atlas)
am = atlas_mask(atlas)
rows = []
for r in tqdm(list(va.sample(min(600, len(va)), random_state=cfg["seed"]).itertuples()), desc="val"):
    gt = load_mask(r.mask_path, size)
    rows.append(dict(class_name=r.class_name, dice=dice(am, gt), iou=iou(am, gt)))
res = pd.DataFrame(rows)
print(f"\nAtlas-only baseline on val: Dice={res.dice.mean():.3f}  IoU={res.iou.mean():.3f}")
print(res.groupby("class_name")[["dice", "iou"]].mean().round(3))
print(f"Lung area fraction of the image (train mean): {np.mean(atlas > 0.5):.3f} (atlas>0.5)")
fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
ax[0].imshow(atlas, cmap="magma"); ax[0].set_title("mean lung mask (train)")
ax[1].imshow(am, cmap="gray"); ax[1].set_title("atlas > 0.5")
for a in ax: a.axis("off")
fig.tight_layout(); fig.savefig(ROOT / cfg["paths"]["reports"] / "figures" / "lung_atlas.png", dpi=130)
print("saved models/lung_atlas.npy")
