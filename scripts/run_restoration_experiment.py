"""Part 2: restoration experiment on simulated degradations (X-rays are rarely blurred, so we
degrade clean images ourselves and measure PSNR/SSIM against the known original).
Usage: python scripts/run_restoration_experiment.py [--n 20]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from tqdm import tqdm
from src.io_utils import ROOT, load_config, load_gray
from src.preprocessing import preprocess
from src.restoration import (gaussian_psf, degrade, inverse_filter, truncated_inverse,
                             wiener_filter, wiener_auto, cls_filter)
from src.evaluation.metrics import psnr, ssim
from src.evaluation.plots import before_after_grid

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20)
args = ap.parse_args()
cfg = load_config()
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
rows = df[df.split == "train"].sample(args.n, random_state=cfg["seed"])
KS = [1e-4, 1e-3, 1e-2, 1e-1]
recs, example = [], None

for r in tqdm(list(rows.itertuples())):
    f = preprocess(load_gray(r.path), cfg["image_size"], "none")
    for bs in (1.0, 2.0, 3.0):
        psf = gaussian_psf(bs)
        for ns in (0.0, 3.0, 10.0):
            g = degrade(f, psf, ns, seed=1)
            outs = {"degraded": g,
                    "inverse": inverse_filter(g, psf),
                    "truncated_inverse": truncated_inverse(g, psf, 0.1),
                    "wiener_auto": wiener_auto(g, psf),
                    "cls": cls_filter(g, psf, 0.01)}
            for K in KS:
                outs[f"wiener_K={K:g}"] = wiener_filter(g, psf, K)
            for name, o in outs.items():
                recs.append(dict(blur_sigma=bs, noise_sigma=ns, method=name,
                                 psnr=psnr(f, o), ssim=ssim(f, o)))
            if example is None and bs == 2.0 and ns == 3.0:
                example = (f, {k: outs[k] for k in ("degraded", "inverse", "truncated_inverse",
                                                    "wiener_auto", "cls")})

res = pd.DataFrame(recs).groupby(["blur_sigma", "noise_sigma", "method"]).mean().reset_index()
res.to_csv(ROOT / cfg["paths"]["reports"] / "restoration_results.csv", index=False)
print(res.pivot_table(index=["blur_sigma", "noise_sigma"], columns="method", values="psnr").round(1).T)
fig_dir = ROOT / cfg["paths"]["reports"] / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
before_after_grid(example[0], example[1], hist=False,
                  title="Restoration (blur sigma=2, noise sigma=3)", path=fig_dir / "restoration_example.png")
