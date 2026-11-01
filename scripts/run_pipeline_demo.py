"""Part 6: run the full pipeline on a few TEST-split images and save sample outputs for the README.
Writes reports/sample_outputs/*.png + sample_predictions.csv, and (with --copy-samples) copies the raw images to
app/samples/ so the dashboard has something to open. These are illustrations, NOT an evaluation.
  python scripts/run_pipeline_demo.py --n 2 --copy-samples"""
import argparse, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.io_utils import ROOT, load_config
from src.pipeline import load_models, analyze

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2, help="images per class")
ap.add_argument("--seg", default="unet")
ap.add_argument("--copy-samples", action="store_true")
ap.add_argument("--models-dir", default=None)
args = ap.parse_args()

cfg = load_config()
models = load_models(args.models_dir, cfg)
for n in models.notes:
    print("NOTE:", n)
df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
te = df[df.split == "test"]
picks = pd.concat([g.sample(min(args.n, len(g)), random_state=cfg["seed"]) for _, g in te.groupby("class_name")])
out = ROOT / cfg["paths"]["reports"] / "sample_outputs"; out.mkdir(parents=True, exist_ok=True)
samples = ROOT / "app" / "samples"
rows = []
for r in picks.itertuples():
    img = cv2.imread(r.path, cv2.IMREAD_UNCHANGED)
    res = analyze(img, models, cfg, seg=args.seg)
    prob = res["prob"]
    pred = "n/a" if prob is None else ("Abnormal" if prob >= 0.5 else "Normal")
    truth = "Normal" if r.label == 0 else "Abnormal"
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for a, im, t in zip(ax, (res["original"], res["enhanced_view"], res["overlay"]),
                        ("input (256 px)", "enhanced (preview)", f"lungs: {res['seg_used']}")):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255); a.set_title(t, fontsize=9); a.axis("off")
    fig.suptitle(f"{r.class_name} | true: {truth} | predicted: {pred}" + ("" if prob is None else f" (p={prob:.2f})"), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{r.class_name.replace(' ', '_')}_{r.image_id.replace(' ', '_')}.png", dpi=130); plt.close(fig)
    rows.append(dict(image_id=r.image_id, class_name=r.class_name, truth=truth, predicted=pred, prob_abnormal=prob,
                     seg=res["seg_used"], lung_area_pct=res["lung_area_pct"]))
    if args.copy_samples:
        samples.mkdir(parents=True, exist_ok=True)
        shutil.copy(r.path, samples / f"{r.class_name.replace(' ', '_')}_{r.image_id.replace(' ', '_')}{Path(r.path).suffix}")
tab = pd.DataFrame(rows)
tab.to_csv(out / "sample_predictions.csv", index=False)
print(tab.round(3).to_string(index=False))
print(f"\nwrote {len(tab)} figures to {out}")
