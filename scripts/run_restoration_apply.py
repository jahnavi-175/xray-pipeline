"""Part 2: decide WHERE restoration is needed on the real data and apply it there.
1) audit every image (noise sigma, Laplacian-variance sharpness)
2) flag the blurriest / noisiest tail (percentiles in config.yaml)
3) Wiener-deblur the flagged-blurry images -> data/restored/ ; save reports/restoration_audit.csv
Then run: python scripts/run_preprocessing.py --restore ...   to apply it inside the main pipeline.
Usage: python scripts/run_restoration_apply.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io_utils import ROOT, load_config, load_gray, save_img
from src.preprocessing import preprocess
from src.restoration import audit_image, gaussian_psf, wiener_filter, blur_score

cfg = load_config()
rc = cfg["restoration"]
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")

imgs, recs = {}, []
for r in tqdm(df.itertuples(), total=len(df), desc="audit"):
    img = preprocess(load_gray(r.path), cfg["image_size"], "none")
    recs.append(dict(image_id=r.image_id, class_name=r.class_name, **audit_image(img)))
aud = pd.DataFrame(recs)
lo, hi = np.percentile(aud.blur_score, rc["blur_percentile"]), np.percentile(aud.sigma_noise, rc["noise_percentile"])
aud["flag_blurry"] = aud.blur_score < lo
aud["flag_noisy"] = aud.sigma_noise > hi
print(aud.groupby("class_name")[["flag_blurry", "flag_noisy"]].mean().round(3))
print(f"flagged blurry: {aud.flag_blurry.sum()} | flagged noisy: {aud.flag_noisy.sum()} of {len(aud)}")

psf = gaussian_psf(rc["psf_sigma"])
flag = df.merge(aud[aud.flag_blurry][["image_id"]], on="image_id")
before_after, gain = [], []
for r in tqdm(flag.itertuples(), total=len(flag), desc="restore"):
    img = preprocess(load_gray(r.path), cfg["image_size"], "none")
    out = wiener_filter(img, psf, rc["K"])
    save_img(ROOT / cfg["paths"]["restored"] / f"{r.image_id}.png", out)
    gain.append(blur_score(out) / max(blur_score(img), 1e-8))
    if len(before_after) < 6:
        before_after.append((img, out))
aud["restored"] = aud.flag_blurry
aud.to_csv(ROOT / cfg["paths"]["reports"] / "restoration_audit.csv", index=False)
if gain:
    print(f"median sharpness gain on restored images: x{np.median(gain):.2f}")
    fig, ax = plt.subplots(2, len(before_after), figsize=(2.8 * len(before_after), 5.6), squeeze=False)
    for j, (a, b) in enumerate(before_after):
        ax[0, j].imshow(a, cmap="gray"); ax[1, j].imshow(b, cmap="gray")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    ax[0, 0].set_title("before", loc="left"); ax[1, 0].set_title("after Wiener", loc="left")
    fig.tight_layout()
    fig.savefig(ROOT / cfg["paths"]["reports"] / "figures" / "restoration_applied.png", dpi=130)
