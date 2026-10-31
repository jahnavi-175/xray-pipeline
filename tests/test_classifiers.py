import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.evaluation.metrics import bootstrap_ci, metrics_row, specificity
from src.classifiers.classical_ml import select_columns, usable_columns
from src.classifiers import run_experiment


def _table(n=300, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "label": y, "mask_ok": 1.0,
                       "class_name": np.where(y == 0, "Normal", rng.choice(["Lung_Opacity", "Viral Pneumonia"], n)),
                       "split": rng.choice(["train", "val", "test"], n, p=[.6, .2, .2])})
    df["glcm_d1_contrast"] = y * signal + rng.normal(size=n)
    df["stat_mean"] = y * signal * 0.5 + rng.normal(size=n)
    for k in range(6):
        df[f"hog_{k:03d}"] = rng.normal(size=n)                  # pure noise columns
    df["shape_const"] = 1.0                                       # constant -> must be dropped
    df["shape_nan"] = np.nan                                      # all NaN -> must be dropped
    df.loc[rng.random(n) < .05, "glcm_d1_contrast"] = np.nan      # some NaNs -> must be imputed
    return df


def test_select_and_usable_columns():
    df = _table()
    cols = select_columns(df, ["shape", "glcm", "stat", "hog"])
    assert "mask_ok" not in cols and "label" not in cols
    use = usable_columns(df[df.split == "train"], cols)
    assert "shape_const" not in use and "shape_nan" not in use and "glcm_d1_contrast" in use


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400); score = y + rng.normal(0, .8, 400); pred = (score > .5).astype(int)
    ci = bootstrap_ci(y, pred, score, n=200)
    acc = (pred == y).mean()
    assert ci["acc"][0] <= acc <= ci["acc"][1] and ci["auc"][0] < ci["auc"][1]


def test_metrics_row_known_values():
    y = np.array([1, 1, 0, 0]); pred = np.array([1, 0, 0, 0])
    r = metrics_row("v_", y, pred, np.array([.9, .4, .2, .1]), ci=False)
    assert r["v_acc"] == 0.75 and r["v_rec"] == 0.5 and r["v_spec"] == 1.0 and r["v_prec"] == 1.0 and r["v_auc"] == 1.0
    assert specificity(y, pred) == 1.0


def test_run_experiment_learns_signal_and_never_touches_test_by_default():
    cfg = {"paths": {"reports": "reports", "features": "data/features"}}
    for kind in ("rf", "svm"):
        m = run_experiment(cfg, "t", "none", "gt", ["shape", "stat", "glcm", "hog"], kind, n_iter=3,
                           tune_n=200, figures=False, log=False, table=_table(), verbose=False)
        assert m["val_auc"] > 0.75 and "test_acc" not in m
        assert m["n_features"] == 8
    m = run_experiment(cfg, "t", "none", "gt", ["glcm", "stat"], "rf", n_iter=2, tune_n=200, final=True,
                       figures=False, log=False, table=_table(), verbose=False)
    assert "test_acc" in m


def test_drop_classes_removes_that_class_from_training_and_evaluation():
    cfg = {"paths": {"reports": "reports", "features": "data/features"}}
    t = _table()
    kw = dict(n_iter=2, tune_n=200, figures=False, log=False, table=t, verbose=False)
    full = run_experiment(cfg, "t", "none", "gt", ["glcm", "stat"], "rf", **kw)
    part = run_experiment(cfg, "t", "none", "gt", ["glcm", "stat"], "rf", drop_classes=("Viral Pneumonia",), **kw)
    assert part["n_train"] < full["n_train"]
    assert not any("Viral" in k for k in part)


def test_svm_without_probability_arg_pca_and_calibrated_save(tmp_path):
    import warnings, joblib
    cfg = {"paths": {"reports": "reports", "features": "data/features"}}
    t = _table()
    out = tmp_path / "clf.joblib"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)               # the deprecation warning must be gone
        m = run_experiment(cfg, "t", "none", "gt", ["glcm", "stat", "hog"], "svm", n_iter=2, tune_n=200, save=True,
                           figures=False, log=False, table=t, verbose=False, svm_pca=3, save_path=out)
    assert m["val_auc"] > 0.7
    bundle = joblib.load(out)
    proba = bundle["pipeline"].predict_proba(t[bundle["columns"]].head(5))
    assert proba.shape == (5, 2) and ((proba >= 0) & (proba <= 1)).all()
