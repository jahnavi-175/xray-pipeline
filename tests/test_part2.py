import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2
from skimage import data
from src.dedup import dhash, near_duplicate_pairs, group_pairs
from src.restoration import blur_score, gaussian_psf, degrade
from src.evaluation.ablation import log_result, load_results

F = cv2.resize(data.camera(), (256, 256), interpolation=cv2.INTER_AREA)


def test_dedup_finds_near_copy_not_different_image():
    noisy = np.clip(F.astype(int) + np.random.default_rng(0).normal(0, 3, F.shape), 0, 255).astype(np.uint8)
    other = cv2.resize(data.coins()[:, :], (256, 256))
    H = np.stack([dhash(F), dhash(noisy), dhash(other)])
    pairs = near_duplicate_pairs(H, 12)
    assert any(a == 0 and b == 1 for a, b, _ in pairs)
    assert not any(b == 2 or a == 2 for a, b, _ in pairs)
    g = group_pairs(3, pairs)
    assert g[0] == g[1] != g[2]


def test_blur_score_orders_sharp_above_blurry():
    assert blur_score(F) > blur_score(degrade(F, gaussian_psf(2.0), 0))


def test_ablation_logger_upserts(tmp_path):
    p = tmp_path / "a.csv"
    log_result(p, "A", {"enhancement": "none"}, {"acc": 0.5})
    log_result(p, "B", {"enhancement": "clahe"}, {"acc": 0.6})
    log_result(p, "A", {"enhancement": "none"}, {"acc": 0.7})
    t = load_results(p)
    assert len(t) == 2 and t.loc[t.config == "A", "acc"].iloc[0] == 0.7
