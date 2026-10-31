"""Part 5: train ONE classifier configuration and (optionally) save it for the dashboard.
  python scripts/train_classifier.py --name F --enh none --masks none_unet --groups shape stat glcm hog --model rf
  ... --final   adds TEST metrics (do this once, at the very end)     ... --save   writes models/classifier.joblib"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.io_utils import load_config
from src.classifiers import run_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="exp")
ap.add_argument("--enh", default="none")
ap.add_argument("--masks", default="none_unet")
ap.add_argument("--groups", nargs="+", default=["shape", "stat", "glcm", "hog"], choices=["shape", "stat", "glcm", "hog"])
ap.add_argument("--model", default="rf", choices=["svm", "rf"])
ap.add_argument("--n-iter", type=int, default=12)
ap.add_argument("--tune-n", type=int, default=4000)
ap.add_argument("--drop-class", nargs="*", default=[], help='e.g. --drop-class "Viral Pneumonia"')
ap.add_argument("--svm-pca", type=int, default=0, help="PCA components before the SVM (0 = off; 100 is much faster)")
ap.add_argument("--final", action="store_true")
ap.add_argument("--save", action="store_true")
args = ap.parse_args()

pd.set_option("display.width", 200)
m = run_experiment(load_config(), args.name, args.enh, args.masks, args.groups, args.model,
                   args.n_iter, args.tune_n, final=args.final, save=args.save, drop_classes=tuple(args.drop_class), svm_pca=args.svm_pca)
for split in ("val", "test"):
    if f"{split}_acc" in m:
        print(f"\n== {split.upper()} ==")
        print("  acc {:.3f} [{:.3f}, {:.3f}]   F1 {:.3f} [{:.3f}, {:.3f}]   AUC {:.3f} [{:.3f}, {:.3f}]".format(
            m[f"{split}_acc"], m[f"{split}_acc_lo"], m[f"{split}_acc_hi"], m[f"{split}_f1"], m[f"{split}_f1_lo"],
            m[f"{split}_f1_hi"], m[f"{split}_auc"], m[f"{split}_auc_lo"], m[f"{split}_auc_hi"]))
        print("  precision {:.3f}  recall {:.3f}  specificity {:.3f}  balanced acc {:.3f}".format(
            m[f"{split}_prec"], m[f"{split}_rec"], m[f"{split}_spec"], m[f"{split}_bal_acc"]))
        print("  per-class correct:", {k.split("hit_")[1]: round(v, 3) for k, v in m.items() if k.startswith(f"{split}_hit_")})
print(f"\ntrain acc {m['train_acc']:.3f}  (a big gap to val = overfitting) | seg Dice {m['seg_dice']:.3f}")
