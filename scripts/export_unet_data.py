"""Pack everything Google Colab needs to train the U-Net into data/colab_bundle/:
   unet_<enh>_<size>.npz  (train/val images + masks, already resized)   and   code.zip (src, scripts, tests, config).
Usage:  python scripts/export_unet_data.py --enh none --size 128           (n-train 0 = use ALL train images)"""
import argparse, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.io_utils import ROOT, load_config
from src.segmentation.unet_data import load_arrays

ap = argparse.ArgumentParser()
ap.add_argument("--enh", default="none")
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--n-train", type=int, default=0, help="0 = all training images")
ap.add_argument("--n-val", type=int, default=500)
args = ap.parse_args()

cfg = load_config()
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
Xtr, Ytr = load_arrays(df[df.split == "train"], cfg, args.enh, args.size, args.n_train or None, cfg["seed"])
Xva, Yva = load_arrays(df[df.split == "val"], cfg, args.enh, args.size, args.n_val, cfg["seed"])

out = ROOT / "data" / "colab_bundle"; out.mkdir(parents=True, exist_ok=True)
npz = out / f"unet_{args.enh}_{args.size}.npz"
np.savez_compressed(npz, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, enh=args.enh, size=args.size)

code = out / "code.zip"                      # forward-slash paths so it unzips cleanly on Linux
with zipfile.ZipFile(code, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ("src", "scripts", "tests"):
        for p in sorted((ROOT / d).rglob("*.py")):
            z.write(p, p.relative_to(ROOT).as_posix())
    z.write(ROOT / "config.yaml", "config.yaml")

print(f"train {len(Xtr)} | val {len(Xva)} | lung fraction {Ytr.mean():.3f}")
for f in (npz, code):
    print(f"{f}  ({f.stat().st_size / 1e6:.1f} MB)")
print("Upload BOTH files to Google Drive (folder 'xray').")
