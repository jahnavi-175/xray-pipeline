"""One ablation experiment = (feature file, feature groups, model) -> tuned fit -> metrics."""
import time
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from ..io_utils import ROOT
from ..evaluation.metrics import metrics_row
from ..evaluation.ablation import log_result
from ..evaluation.seg_eval import segmentation_scores
from .classical_ml import select_columns, usable_columns, build_pipeline, tune, get_scores

# interpretable features the dashboard compares against typical Normal / Abnormal training values
INTERPRETABLE = ["shape_area_frac", "shape_medial_gap_ratio", "shape_lr_area_ratio", "stat_mean", "stat_std",
                 "stat_lung_to_rest", "stat_lr_mean_diff", "glcm_d1_entropy", "glcm_d1_contrast"]
COLORS = {"shape": "#4c72b0", "stat": "#55a868", "glcm": "#c44e52", "hog": "#8172b2"}


def feature_path(cfg, enh, masks):
    return ROOT / cfg["paths"]["features"] / f"features_{enh}__{masks}.csv"


def _figures(name, kind, pipe, cols, va, pred, score, rep):
    fig_dir = rep / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ConfusionMatrixDisplay.from_predictions(va.label, pred, display_labels=["Normal", "Abnormal"], ax=ax[0], colorbar=False)
    RocCurveDisplay.from_predictions(va.label, score, ax=ax[1]); ax[1].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set_title(f"{name} / {kind}: confusion (val)"); ax[1].set_title("ROC (val)")
    fig.tight_layout(); fig.savefig(fig_dir / f"clf_{name}_{kind}_cm_roc.png", dpi=130); plt.close(fig)
    if kind == "rf":
        imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=cols).sort_values().tail(20)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(imp.index, imp.values, color=[COLORS[c.split("_")[0]] for c in imp.index])
        ax.set_title(f"{name}: top-20 RF importances"); fig.tight_layout()
        fig.savefig(fig_dir / f"clf_{name}_rf_importance.png", dpi=130); plt.close(fig)


def run_experiment(cfg, name, enh, masks, groups, kind, n_iter=12, tune_n=4000, final=False, save=False,
                   figures=True, log=True, table=None, verbose=True, seed=42, drop_classes=(), svm_pca=0, save_path=None):
    """Returns a dict of metrics. Decisions use VAL; pass final=True ONCE at the end to add TEST metrics."""
    df = pd.read_csv(feature_path(cfg, enh, masks)) if table is None else table
    if drop_classes:                       # e.g. ("Viral Pneumonia",): test for dataset-source shortcuts
        df = df[~df.class_name.isin(drop_classes)]
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    cols = usable_columns(tr, select_columns(df, groups))
    t0 = time.time()
    best, cv_auc = tune(kind, tr[cols], tr.label, n_iter, tune_n, seed=seed, pca=svm_pca)
    pipe = build_pipeline(kind, seed=seed, pca=svm_pca)
    pipe.set_params(**best)
    if save and kind == "svm":        # probabilities for the dashboard: sigmoid-calibrated SVM
        pipe.steps[-1] = ("clf", CalibratedClassifierCV(pipe.steps[-1][1], method="sigmoid", cv=5, ensemble=False))
    pipe.fit(tr[cols], tr.label)
    fit_min = (time.time() - t0) / 60

    m = {"n_features": len(cols), "n_train": len(tr), "cv_auc": cv_auc}
    sub = tr.sample(min(3000, len(tr)), random_state=seed)
    m["train_acc"] = float((pipe.predict(sub[cols]) == sub.label).mean())
    splits = [("val", va)] + ([("test", te)] if final else [])
    for sname, part in splits:
        pred, score = pipe.predict(part[cols]), get_scores(pipe, part[cols])
        m.update(metrics_row(sname + "_", part.label.values, pred, score))
        for cname, g in part.groupby("class_name"):                  # which subtype is missed?
            hit = (pred[part.class_name.values == cname] == (0 if g.label.iloc[0] == 0 else 1)).mean()
            m[f"{sname}_hit_{cname.replace(' ', '_')}"] = float(hit)
        if sname == "val":
            vpred, vscore = pred, score
    if figures:
        _figures(name, kind, pipe, cols, va, vpred, vscore, ROOT / cfg["paths"]["reports"])
    sd, si = segmentation_scores(cfg, masks)
    m["seg_dice"], m["seg_iou"] = sd, si
    if log:
        params = dict(enhancement=enh, restoration="wiener(flagged)", segmentation=masks,
                      features="+".join(groups), classifier=kind,
                      dropped_classes="+".join(drop_classes))
        log_result(ROOT / cfg["paths"]["reports"] / "ablation_table.csv", f"{name}-{kind}", params, m)
    if save:
        out = save_path or ROOT / "models" / "classifier.joblib"
        out.parent.mkdir(exist_ok=True)
        ref = {c: {"normal": float(tr[tr.label == 0][c].median()), "abnormal": float(tr[tr.label == 1][c].median())}
               for c in INTERPRETABLE if c in cols}
        joblib.dump({"pipeline": pipe, "columns": cols, "enh": enh, "masks": masks, "groups": list(groups),
                     "kind": kind, "params": best, "svm_pca": svm_pca, "reference": ref,
                     "val_scores": np.asarray(vscore, dtype=np.float32), "val_labels": va.label.values.astype(int)}, out)
        if verbose:
            print("saved", out)
    if verbose:
        print(f"[{name}-{kind}] features={len(cols)} best={ {k: (round(v, 5) if isinstance(v, float) else v) for k, v in best.items()} } "
              f"cv_auc={cv_auc:.3f} ({fit_min:.1f} min)")
    return m
