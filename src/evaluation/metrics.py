"""Segmentation, restoration and classification metrics."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)


def dice(pred, gt, eps=1e-8):
    p, g = pred.astype(bool), gt.astype(bool)
    return float((2 * (p & g).sum() + eps) / (p.sum() + g.sum() + eps))


def iou(pred, gt, eps=1e-8):
    p, g = pred.astype(bool), gt.astype(bool)
    return float(((p & g).sum() + eps) / ((p | g).sum() + eps))


def psnr(ref, img):
    return float(peak_signal_noise_ratio(ref, img, data_range=255))


def ssim(ref, img):
    return float(structural_similarity(ref, img, data_range=255))


def classification_summary(y_true, y_pred, y_prob=None):
    out = dict(accuracy=accuracy_score(y_true, y_pred),
               precision=precision_score(y_true, y_pred, zero_division=0),
               recall=recall_score(y_true, y_pred, zero_division=0),
               f1=f1_score(y_true, y_pred, zero_division=0))
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return out


def region_separability(img, mask):
    """Contrast between inside and outside of a mask (d'-like): |mu_in-mu_out| / pooled std.
    Higher = region of interest is easier to separate by intensity (helps thresholding)."""
    m = mask.astype(bool)
    if m.sum() < 10 or (~m).sum() < 10:
        return float("nan")
    a, b = img[m].astype(np.float64), img[~m].astype(np.float64)
    return float(abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-8))


def specificity(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp = ((y == 0) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def bootstrap_ci(y, pred, score=None, n=500, seed=0, alpha=0.05):
    """Percentile bootstrap CIs for accuracy, F1 and (if score is given) ROC-AUC.
    Two configs whose CIs overlap heavily are NOT meaningfully different."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    idx = rng.integers(0, len(y), (n, len(y)))
    yy, pp = y[idx], pred[idx]
    acc = (yy == pp).mean(axis=1)
    tp = ((yy == 1) & (pp == 1)).sum(axis=1); fp = ((yy == 0) & (pp == 1)).sum(axis=1)
    fn = ((yy == 1) & (pp == 0)).sum(axis=1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    out = {"acc": acc, "f1": f1}
    if score is not None:
        score = np.asarray(score)
        aucs = []
        for k in range(n):
            if len(np.unique(yy[k])) == 2:
                aucs.append(roc_auc_score(yy[k], score[idx[k]]))
        out["auc"] = np.array(aucs)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {k: tuple(np.percentile(v, q)) for k, v in out.items()}


def metrics_row(prefix, y, pred, score=None, ci=True):
    """Flat dict of metrics with a prefix (e.g. 'val_'), ready for the ablation table."""
    y, pred = np.asarray(y), np.asarray(pred)
    rec = recall_score(y, pred, zero_division=0)
    spec = specificity(y, pred)
    row = {"acc": accuracy_score(y, pred), "prec": precision_score(y, pred, zero_division=0), "rec": rec,
           "spec": spec, "f1": f1_score(y, pred, zero_division=0), "bal_acc": (rec + spec) / 2}
    if score is not None:
        row["auc"] = roc_auc_score(y, score)
    if ci:
        for k, (lo, hi) in bootstrap_ci(y, pred, score).items():
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
    return {prefix + k: float(v) for k, v in row.items()}
