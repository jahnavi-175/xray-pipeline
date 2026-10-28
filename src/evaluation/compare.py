"""Experiment helpers shared by scripts and notebooks (Part 1 comparisons)."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..io_utils import load_gray, load_mask
from ..preprocessing import preprocess, denoise
from ..enhancement import enhance, quality_report
from ..noise_models import add_gaussian, add_salt_pepper, add_speckle, add_poisson
from .metrics import region_separability, psnr, ssim


def sample_rows(df, n_per_class, seed=0):
    parts = [g.sample(min(n_per_class, len(g)), random_state=seed) for _, g in df.groupby("class_name")]
    return pd.concat(parts)


def load_pre(row, cfg):
    pp = cfg["preprocessing"]
    return preprocess(load_gray(row.path), cfg["image_size"], pp["denoise"], pp["denoise_ksize"])


def compare_enhancements(df, cfg, methods, n_per_class=30, seed=0):
    """Per-image quality (entropy, contrast, sharpness) + lung/background separability."""
    rows = sample_rows(df, n_per_class, seed)
    ref_row = df[df.label == 0].iloc[0]
    reference = load_pre(ref_row, cfg)
    recs = []
    for r in tqdm(list(rows.itertuples()), desc="enhancement metrics"):
        img = load_pre(r, cfg)
        mask = load_mask(r.mask_path, cfg["image_size"]) if r.mask_path else None
        for m in methods:
            out = enhance(img, m, cfg["enhancement"], reference)
            rec = dict(image_id=r.image_id, class_name=r.class_name, method=m, **quality_report(out))
            rec["separability"] = region_separability(out, mask) if mask is not None else np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


NOISES = {
    "gaussian(s=15)": lambda x: add_gaussian(x, 15),
    "salt&pepper(3%)": lambda x: add_salt_pepper(x, 0.03),
    "speckle(0.15)": lambda x: add_speckle(x, 0.15),
    "poisson": lambda x: add_poisson(x),
}
DENOISERS = {
    "none": lambda x: x,
    "median3": lambda x: denoise(x, "median", 3),
    "median5": lambda x: denoise(x, "median", 5),
    "gaussian5": lambda x: denoise(x, "gaussian", 5),
    "bilateral": lambda x: denoise(x, "bilateral"),
    "nlm": lambda x: denoise(x, "nlm"),
}


def denoise_benchmark(df, cfg, n=20, seed=0):
    """Which denoiser suits which noise model? PSNR/SSIM against the clean image."""
    rows = df.sample(min(n, len(df)), random_state=seed)
    recs = []
    for r in tqdm(list(rows.itertuples()), desc="denoise benchmark"):
        clean = preprocess(load_gray(r.path), cfg["image_size"], "none")
        for nn, nf in NOISES.items():
            noisy = nf(clean)
            for dn, df_ in DENOISERS.items():
                out = df_(noisy)
                recs.append(dict(noise=nn, denoiser=dn, psnr=psnr(clean, out), ssim=ssim(clean, out)))
    t = pd.DataFrame(recs).groupby(["noise", "denoiser"]).mean().reset_index()
    return t
