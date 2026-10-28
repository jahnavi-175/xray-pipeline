import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.io_utils import ROOT, load_config, load_gray

METHODS = ["clahe", "he", "histspec", "homomorphic", "highpass"]

cfg = load_config()
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
folders = {"processed": ROOT / cfg["paths"]["processed"],
           **{m: ROOT / cfg["paths"]["enhanced"] / m for m in METHODS}}

for name, d in folders.items():
    n = len(list(d.glob("*.png")))
    print(f"{name:12s} {n} files (expected {len(df)})", "OK" if n == len(df) else "MISMATCH")

ids = df.sample(4, random_state=1).image_id
ncols = len(folders)
fig, ax = plt.subplots(len(ids), ncols, figsize=(2.6 * ncols, 2.7 * len(ids)))
for i, iid in enumerate(ids):
    for j, (name, d) in enumerate(folders.items()):
        im = load_gray(d / f"{iid}.png")
        assert im.shape == (cfg["image_size"],) * 2, (name, iid, im.shape)
        ax[i, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[i, j].axis("off")
        if i == 0:
            ax[i, j].set_title(name, fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "reports/figures/final_check.png", dpi=120)
print("wrote reports/figures/final_check.png")