"""Part 3: evaluate classical lung segmentation on every enhancement variant, and write masks.

First (once):  python scripts/build_atlas.py     # mean lung shape from TRAIN masks -> models/lung_atlas.npy
Evaluate (uses the VALIDATION split, never test):
    python scripts/run_segmentation.py --n 150      # each method plain AND with the atlas prior ("+prior")
Write predicted masks for ALL images with the chosen combination (needed for Part 4):
    python scripts/run_segmentation.py --write clahe kmeans
Outputs: reports/segmentation_results.csv, segmentation_summary.csv, figures/segmentation_*.png"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io_utils import ROOT, load_config, load_gray, load_mask, save_img
from src.segmentation import segment_lungs, METHODS, load_atlas, atlas_mask
from src.evaluation.metrics import dice, iou
from src.evaluation.compare import sample_rows

ENH_ALL = ["none", "he", "clahe", "histspec", "homomorphic", "highpass"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150, help="images per class from the val split")
ap.add_argument("--enh", nargs="+", default=ENH_ALL)
ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
ap.add_argument("--split", default="val")
ap.add_argument("--write", nargs=2, metavar=("ENH", "METHOD"), help="METHOD may end in +prior")
ap.add_argument("--no-prior", action="store_true")
ap.add_argument("--unet", default=None, help="path to a trained U-Net checkpoint (adds a 'unet' column)")
args = ap.parse_args()

pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
cfg = load_config()
size = cfg["image_size"]
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")


def img_path(enh, image_id):
    base = ROOT / cfg["paths"]["processed"] if enh == "none" else ROOT / cfg["paths"]["enhanced"] / enh
    return base / f"{image_id}.png"


ATLAS = load_atlas()
UNET = None


def get_unet(path):
    global UNET
    if UNET is None:
        from src.segmentation.unet import UNetSegmenter     # imports PyTorch only when needed
        UNET = UNetSegmenter.load(path)
    return UNET


def run_seg(img, name):
    """name = method, method+prior or unet"""
    if name == "unet":
        return get_unet(args.unet)(img)
    if name.endswith("+prior"):
        return segment_lungs(img, name[:-6], prior=ATLAS)
    return segment_lungs(img, name)


def overlay(img, gt, pred):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for m, col in ((gt, (0, 255, 0)), (pred, (255, 0, 0))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cs, -1, col, 1)
    return rgb


if args.write:                                  # ---------- mask export for all images ----------
    enh, method = args.write
    if method == "unet" and args.unet is None:
        args.unet = str(ROOT / "models" / f"unet_{enh}.pt")
    out_dir = ROOT / cfg["paths"]["predicted_masks"] / f"{enh}_{method.replace('+', '_')}"
    for r in tqdm(df.itertuples(), total=len(df), desc=f"writing {enh}_{method}"):
        pred = run_seg(load_gray(img_path(enh, r.image_id)), method)
        save_img(out_dir / f"{r.image_id}.png", pred * 255)
    print("wrote", out_dir)
    sys.exit(0)

sub = df[(df.split == args.split) & df.mask_path.notna()]
rows = sample_rows(sub, args.n, cfg["seed"])
gts = {r.image_id: load_mask(r.mask_path, size) for r in rows.itertuples()}
names = list(args.methods)
unet_enh = None
if args.unet:
    unet_enh = get_unet(args.unet).enh
    names.append("unet")
    print(f"U-Net loaded (trained on enhancement '{unet_enh}'): it is evaluated on that variant only.")
if ATLAS is not None and not args.no_prior:
    names += [m + "+prior" for m in args.methods]
elif ATLAS is None:
    print("NOTE: models/lung_atlas.npy not found -> run scripts/build_atlas.py to also evaluate the prior variants.")
else:
    pass
if ATLAS is not None:
    am = atlas_mask(ATLAS)
    base = np.mean([dice(am, g) for g in gts.values()])
    print(f"\nAtlas-only baseline (no image used): Dice = {base:.3f}  <- methods must beat this")
recs = []
for enh in args.enh:
    for r in tqdm(list(rows.itertuples()), desc=enh):
        img = load_gray(img_path(enh, r.image_id))
        for m in names:
            if m == "unet" and enh != unet_enh:
                continue
            pred = run_seg(img, m)
            recs.append(dict(image_id=r.image_id, class_name=r.class_name, enh=enh, seg=m,
                             dice=dice(pred, gts[r.image_id]), iou=iou(pred, gts[r.image_id]),
                             empty=int(pred.sum() == 0)))
res = pd.DataFrame(recs)
rep = ROOT / cfg["paths"]["reports"]; (rep / "figures").mkdir(parents=True, exist_ok=True)
res.to_csv(rep / "segmentation_results.csv", index=False)
summ = res.groupby(["enh", "seg"])[["dice", "iou", "empty"]].agg(
    dice=("dice", "mean"), dice_std=("dice", "std"), iou=("iou", "mean"), empty_masks=("empty", "sum")).round(3)
summ.to_csv(rep / "segmentation_summary.csv")
print("\nMean Dice (rows = enhancement, cols = segmentation):")
print(res.pivot_table(index="enh", columns="seg", values="dice").round(3))
print("\nBy class (mean Dice):")
print(res.pivot_table(index="class_name", columns="seg", values="dice").round(3))
print("\nEmpty masks:", int(res["empty"].sum()), "of", len(res))

# heatmap
pv = res.pivot_table(index="enh", columns="seg", values="dice")
fig, ax = plt.subplots(figsize=(1.0 * pv.shape[1] + 3, 4))
im = ax.imshow(pv.values, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(pv.shape[1]), pv.columns, rotation=30); ax.set_yticks(range(pv.shape[0]), pv.index)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
plt.colorbar(im); ax.set_title("Mean Dice on validation images"); fig.tight_layout()
fig.savefig(rep / "figures" / "segmentation_heatmap.png", dpi=150)

# examples + worst failures for the best combination
best = summ.dice.idxmax(); enh, m = best
print(f"\nBest combination: enhancement={enh}, method={m}, Dice={summ.dice.max():.3f}")
sel = res[(res.enh == enh) & (res.seg == m)].sort_values("dice")
for name, picks in (("failures", sel.head(6)), ("examples", sel.iloc[len(sel) // 2:].sample(6, random_state=0))):
    fig, ax = plt.subplots(1, 6, figsize=(16, 3.2))
    for a, r in zip(ax, picks.itertuples()):
        img = load_gray(img_path(enh, r.image_id))
        a.imshow(overlay(img, gts[r.image_id], run_seg(img, m)))
        a.set_title(f"{r.class_name[:6]} dice={r.dice:.2f}", fontsize=8); a.axis("off")
    fig.suptitle(f"{enh} + {m}: {name} (green = ground truth, red = prediction)")
    fig.tight_layout(); fig.savefig(rep / "figures" / f"segmentation_{name}.png", dpi=130)
