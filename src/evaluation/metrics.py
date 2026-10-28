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
