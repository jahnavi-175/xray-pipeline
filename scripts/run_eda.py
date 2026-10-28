"""Part 1: dataset study. Writes reports/eda_summary.csv, splits CSV and figures.
Usage (from repo root):  python scripts/run_eda.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io_utils import ROOT, load_config, build_index, load_gray
from src.noise_models import estimate_sigma
from src.enhancement import quality_report

cfg = load_config()
df = build_index(cfg)
fig_dir = ROOT / cfg["paths"]["reports"] / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

rows = []
for r in tqdm(df.itertuples(), total=len(df)):
    img = load_gray(r.path)
    q = quality_report(img)
    rows.append(dict(image_id=r.image_id, h=img.shape[0], w=img.shape[1], mean=img.mean(),
                     std=img.std(), sigma_noise=estimate_sigma(img), **q))
stats = df.merge(pd.DataFrame(rows), on="image_id")
stats.to_csv(ROOT / cfg["paths"]["reports"] / "eda_summary.csv", index=False)
(ROOT / cfg["paths"]["features"]).mkdir(parents=True, exist_ok=True)
df.to_csv(ROOT / cfg["paths"]["features"] / "splits.csv", index=False)

print(stats.groupby("class_name")[["h", "w", "mean", "std", "sigma_noise", "entropy"]].describe().T)
print("\nClass balance:\n", df["label"].value_counts(), "\nSplits:\n", df["split"].value_counts())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df["class_name"].value_counts().plot.bar(ax=ax[0], title="Class distribution")
for c, g in stats.groupby("class_name"):
    ax[1].hist(g["mean"], bins=40, alpha=.5, label=c)
    ax[2].hist(g["sigma_noise"], bins=40, alpha=.5, label=c)
ax[1].set_title("Mean intensity"); ax[2].set_title("Estimated noise sigma"); ax[1].legend()
plt.tight_layout(); plt.savefig(fig_dir / "eda_overview.png", dpi=150)

# sample grid per class for visual label/artifact inspection
classes = df["class_name"].unique()
fig, ax = plt.subplots(len(classes), 6, figsize=(15, 2.5 * len(classes)))
for i, c in enumerate(classes):
    for j, p in enumerate(df[df.class_name == c].sample(6, random_state=0)["path"]):
        ax[i, j].imshow(load_gray(p), cmap="gray"); ax[i, j].axis("off")
    ax[i, 0].set_title(c, loc="left")
plt.tight_layout(); plt.savefig(fig_dir / "eda_samples.png", dpi=120)
