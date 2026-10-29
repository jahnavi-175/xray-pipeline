"""Part 4: which features separate the classes? Univariate ROC-AUC on the TRAIN split only.
Usage: python scripts/analyze_features.py --file data/features/features_none__none_unet.csv"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.io_utils import ROOT, load_config
from src.features import groups_of

ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True)
args = ap.parse_args()
cfg = load_config()
tab = pd.read_csv(args.file)
tr = tab[(tab.split == "train") & (tab.mask_ok == 1)]
groups = groups_of(tab.columns)
rows = []
for g, cols in groups.items():
    for c in cols:
        x = tr[c]
        if x.isna().mean() > 0.2 or x.nunique() < 3:
            continue
        a = roc_auc_score(tr.label, x.fillna(x.median()))
        rows.append(dict(group=g, feature=c, auc=a, strength=abs(a - 0.5) + 0.5))
res = pd.DataFrame(rows).sort_values("strength", ascending=False)
name = Path(args.file).stem
res.to_csv(ROOT / cfg["paths"]["reports"] / f"feature_auc_{name}.csv", index=False)
print("Top 15 single features (AUC; direction ignored):")
print(res.head(15)[["group", "feature", "strength"]].round(3).to_string(index=False))
print("\nBest / median per group:")
print(res.groupby("group").strength.agg(["max", "median", "count"]).round(3))
corr = tr[[c for g in ("shape", "stat", "glcm") for c in groups[g]]].corr().abs()
hi = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)) > 0.95).sum().sum()
print(f"\nhighly redundant handcrafted pairs (|r|>0.95): {hi}")
top = res.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
col = {"shape": "#4c72b0", "stat": "#55a868", "glcm": "#c44e52", "hog": "#8172b2"}
ax.barh(top.feature, top.strength - 0.5, left=0.5, color=[col[g] for g in top.group])
ax.set_xlabel("single-feature AUC (train)"); ax.set_title("Top 20 features")
fig.tight_layout(); fig.savefig(ROOT / cfg["paths"]["reports"] / "figures" / f"feature_auc_{name}.png", dpi=130)
