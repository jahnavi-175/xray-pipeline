"""Pack images for training the CNN on Google Colab: data/colab_bundle/cnn_<enh>_<size>.npz + code.zip.
Usage:  python scripts/export_cnn_data.py --enh none --size 128"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io_utils import ROOT, load_config, load_gray, enhanced_path
from src.utils import make_code_zip

ap = argparse.ArgumentParser()
ap.add_argument("--enh", default="none")
ap.add_argument("--size", type=int, default=128)
args = ap.parse_args()
cfg = load_config()
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
X = np.stack([cv2.resize(load_gray(enhanced_path(cfg, args.enh, r.image_id)), (args.size, args.size),
                         interpolation=cv2.INTER_AREA) for r in tqdm(list(df.itertuples()))])
out = ROOT / "data" / "colab_bundle"; out.mkdir(parents=True, exist_ok=True)
npz = out / f"cnn_{args.enh}_{args.size}.npz"
np.savez_compressed(npz, X=X, y=df.label.values.astype(np.int64), split=np.array(df.split.tolist(), dtype="U"),
                    cls=np.array(df.class_name.tolist(), dtype="U"), enh=args.enh, size=args.size)
make_code_zip(ROOT, out / "code.zip")
print(f"{len(X)} images | {npz.name}: {npz.stat().st_size / 1e6:.0f} MB | code.zip refreshed")
print("Upload both files from data/colab_bundle/ to your Google Drive folder 'xray' (replace the old code.zip).")
