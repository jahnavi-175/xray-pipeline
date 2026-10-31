"""Part 5/6: run the ablation grid on every feature file that exists and print the table.
  python scripts/run_ablation.py --models rf          # fast first pass
  python scripts/run_ablation.py --models svm rf      # full
  python scripts/run_ablation.py --cnn reports/cnn_none_128.json --cnn-name I_cnn_none   # add a CNN row"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.io_utils import ROOT, load_config
from src.classifiers import run_experiment
from src.classifiers.experiment import feature_path
from src.evaluation.ablation import log_result

ALL = ["shape", "stat", "glcm", "hog"]
CONFIGS = [   # name, enhancement, masks, groups
    ("A_noseg",        "none",  "full",      ["stat", "glcm", "hog"]),
    ("K_he_all",          "he",          "none_unet", ALL),
    ("L_histspec_all",    "histspec",    "none_unet", ALL),
    ("M_homomorphic_all", "homomorphic", "none_unet", ALL),
    ("N_highpass_all",    "highpass",    "none_unet", ALL),
    ("A2_unet_nosh",   "none",  "none_unet", ["stat", "glcm", "hog"]),   # same features as A, but with masks
    ("B_unet_shape",   "none",  "none_unet", ["shape"]),
    ("C_unet_stat",    "none",  "none_unet", ["stat"]),
    ("D_unet_glcm",    "none",  "none_unet", ["glcm"]),
    ("E_unet_hog",     "none",  "none_unet", ["hog"]),
    ("F_unet_hand",    "none",  "none_unet", ["shape", "stat", "glcm"]),
    ("G_unet_all",     "none",  "none_unet", ALL),
    ("H_clahe_all",    "clahe", "none_unet", ALL),
    ("I_gt_all",       "none",  "gt",        ALL),
]
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["rf"], choices=["svm", "rf"])
ap.add_argument("--n-iter", type=int, default=12)
ap.add_argument("--tune-n", type=int, default=4000)
ap.add_argument("--svm-pca", type=int, default=0, help="PCA components before the SVM (0 = off; 100 is much faster)")
ap.add_argument("--only", nargs="*", default=None, help="config name prefixes, e.g. G H")
ap.add_argument("--cnn", default=None); ap.add_argument("--cnn-name", default="J_cnn")
args = ap.parse_args()
cfg = load_config()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)

if args.cnn:
    j = json.load(open(args.cnn))
    log_result(ROOT / cfg["paths"]["reports"] / "ablation_table.csv", args.cnn_name,
               dict(enhancement=j["enh"], restoration="wiener(flagged)", segmentation="none (end-to-end)",
                    features="learned", classifier="cnn-" + j["arch"]), j["metrics"])
    print("logged CNN row", args.cnn_name)
else:
    for name, enh, masks, groups in CONFIGS:
        if args.only and not any(name.startswith(p) for p in args.only):
            continue
        if not feature_path(cfg, enh, masks).exists():
            print(f"skip {name}: {feature_path(cfg, enh, masks).name} not found"); continue
        for kind in args.models:
            run_experiment(cfg, name, enh, masks, groups, kind, args.n_iter, args.tune_n, svm_pca=args.svm_pca)

t = pd.read_csv(ROOT / cfg["paths"]["reports"] / "ablation_table.csv")
cols = ["config", "enhancement", "segmentation", "seg_dice", "features", "classifier", "n_features",
        "val_acc", "val_f1", "val_auc", "val_rec", "val_spec", "train_acc"]
show = t[[c for c in cols if c in t.columns]].round(3)
print("\n", show.to_string(index=False))
show.to_csv(ROOT / cfg["paths"]["reports"] / "ablation_summary.csv", index=False)
try:
    md = show.to_markdown(index=False)          # needs the optional 'tabulate' package
except ImportError:
    md = show.to_string(index=False)
(ROOT / cfg["paths"]["reports"] / "ablation_summary.md").write_text(md)
