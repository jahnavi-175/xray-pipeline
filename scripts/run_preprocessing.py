"""Part 1: preprocess every image and write one enhanced copy per method.
Usage:  python scripts/run_preprocessing.py --methods clahe homomorphic highpass
Output: data/processed/<id>.png and data/enhanced/<method>/<id>.png"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm
from src.io_utils import ROOT, load_config, load_gray, save_img
from src.preprocessing import preprocess
from src.enhancement import enhance, METHODS
from src.restoration import gaussian_psf, wiener_filter

ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=["clahe", "homomorphic", "highpass"], choices=METHODS)
ap.add_argument("--ref", default=None, help="reference image id for histspec")
ap.add_argument("--restore", action="store_true", help="Wiener-deblur images flagged in reports/restoration_audit.csv")
args = ap.parse_args()

cfg = load_config()
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")  # created by run_eda.py
size, pp = cfg["image_size"], cfg["preprocessing"]
restore_ids, psf = set(), gaussian_psf(cfg["restoration"]["psf_sigma"])
if args.restore:
    aud = pd.read_csv(ROOT / cfg["paths"]["reports"] / "restoration_audit.csv")
    restore_ids = set(aud[aud.flag_blurry].image_id)
    print(f"restoring {len(restore_ids)} flagged images")

reference = None
if "histspec" in args.methods:
    ref_row = df[df.image_id == args.ref].iloc[0] if args.ref else df[df.label == 0].iloc[0]
    reference = preprocess(load_gray(ref_row.path), size, "none")

for r in tqdm(df.itertuples(), total=len(df)):
    img = preprocess(load_gray(r.path), size, pp["denoise"], pp["denoise_ksize"])
    if r.image_id in restore_ids:
        img = wiener_filter(img, psf, cfg["restoration"]["K"])
    save_img(ROOT / cfg["paths"]["processed"] / f"{r.image_id}.png", img)
    for m in args.methods:
        out = enhance(img, m, cfg["enhancement"], reference)
        save_img(ROOT / cfg["paths"]["enhanced"] / m / f"{r.image_id}.png", out)
