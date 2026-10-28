"""Part 1 quantitative comparison. Usage: python scripts/run_enhancement_comparison.py [--n 30]
Writes reports/enhancement_metrics.csv, enhancement_summary.csv and denoise_benchmark.csv"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.io_utils import ROOT, load_config
from src.enhancement import METHODS
from src.evaluation.compare import compare_enhancements, denoise_benchmark

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30)
args = ap.parse_args()
cfg = load_config()
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
df = df[df.split == "train"]               # tune/inspect on train only
out = ROOT / cfg["paths"]["reports"]

res = compare_enhancements(df, cfg, METHODS, args.n)
res.to_csv(out / "enhancement_metrics.csv", index=False)
summary = res.groupby("method")[["entropy", "contrast", "sharpness", "separability"]].agg(["mean", "std"]).round(3)
summary.to_csv(out / "enhancement_summary.csv")
print(summary)
t = denoise_benchmark(df, cfg, 20)
t.to_csv(out / "denoise_benchmark.csv", index=False)
print(t.pivot(index="denoiser", columns="noise", values="psnr").round(2))
