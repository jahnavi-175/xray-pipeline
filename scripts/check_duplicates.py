"""Part 2: leakage check. Finds near-duplicate images, reports pairs that straddle train/val/test,
and with --fix moves each duplicate group into a single split (written back to splits.csv).
Usage: python scripts/check_duplicates.py [--fix] [--max-dist 12]"""
import argparse, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io_utils import ROOT, load_config, load_gray
from src.dedup import dhash, near_duplicate_pairs, group_pairs

ap = argparse.ArgumentParser()
ap.add_argument("--fix", action="store_true")
ap.add_argument("--max-dist", type=int, default=None)
args = ap.parse_args()
cfg = load_config()
max_dist = args.max_dist if args.max_dist is not None else cfg["dedup"]["max_dist"]
sp_path = ROOT / cfg["paths"]["features"] / "splits.csv"
df = pd.read_csv(sp_path)

H = np.stack([dhash(load_gray(p), cfg["dedup"]["hash_size"]) for p in tqdm(df.path, desc="hashing")])
pairs = near_duplicate_pairs(H, max_dist)
groups = group_pairs(len(df), pairs)
df["dup_group"] = groups
sizes = pd.Series(groups).value_counts()
n_in_dups = int((pd.Series(groups).map(sizes) > 1).sum())
cross = [(a, b, d) for a, b, d in pairs if df.split[a] != df.split[b]]
label_clash = [(a, b, d) for a, b, d in pairs if df.label[a] != df.label[b]]
print(f"images: {len(df)} | near-duplicate pairs: {len(pairs)} | images in a dup group: {n_in_dups}")
print(f"pairs crossing splits (LEAKAGE): {len(cross)} | pairs with conflicting labels: {len(label_clash)}")

rep = pd.DataFrame([dict(id_a=df.image_id[a], id_b=df.image_id[b], dist=d, split_a=df.split[a],
                         split_b=df.split[b], label_a=df.label[a], label_b=df.label[b]) for a, b, d in pairs])
rep.to_csv(ROOT / cfg["paths"]["reports"] / "duplicate_pairs.csv", index=False)

# contact sheet of the closest cross-split pairs for visual verification (check for false positives!)
show = sorted(cross, key=lambda t: t[2])[:6]
if show:
    fig, ax = plt.subplots(2, len(show), figsize=(3 * len(show), 6), squeeze=False)
    for j, (a, b, d) in enumerate(show):
        for i, k in enumerate((a, b)):
            ax[i, j].imshow(load_gray(df.path[k]), cmap="gray"); ax[i, j].axis("off")
            ax[i, j].set_title(f"{df.split[k]} | d={d}", fontsize=9)
    fig.tight_layout()
    out = ROOT / cfg["paths"]["reports"] / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "duplicates_check.png", dpi=120)
    print("Wrote reports/figures/duplicates_check.png - LOOK at it before using --fix.")

if args.fix and cross:
    for g, members in df.groupby("dup_group").groups.items():
        if len(members) > 1:
            win = Counter(df.loc[members, "split"]).most_common(1)[0][0]
            df.loc[members, "split"] = win
    df.drop(columns="dup_group").to_csv(sp_path, index=False)
    print("splits.csv updated.", df.split.value_counts().to_dict())
elif args.fix:
    print("Nothing to fix.")
