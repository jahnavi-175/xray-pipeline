import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from tests.phantom import make_phantom
from src.segmentation.unet_data import augment, resize_pair


def test_augment_keeps_shape_dtype_and_binary_mask():
    img, gt = make_phantom(0, variant="tight")
    a, m = augment(img, gt, np.random.default_rng(0))
    assert a.shape == img.shape and a.dtype == np.uint8
    assert m.shape == gt.shape and set(np.unique(m)) <= {0, 1}
    assert 0.3 * gt.sum() < m.sum() < 3 * gt.sum()          # lungs neither vanish nor explode


def test_resize_pair_binary():
    img, gt = make_phantom(0)
    x, y = resize_pair(img, gt, 128)
    assert x.shape == (128, 128) and set(np.unique(y)) <= {0, 1}


def test_unet_forward_shape_and_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from src.segmentation.unet import UNet, save_checkpoint, UNetSegmenter, dice_loss
    m = UNet(base=4)
    x = torch.rand(2, 1, 64, 64)
    y = m(x)
    assert y.shape == (2, 1, 64, 64)
    loss = dice_loss(y, (torch.rand(2, 1, 64, 64) > .5).float())
    loss.backward()
    assert torch.isfinite(loss)
    p = tmp_path / "u.pt"
    save_checkpoint(p, m, 4, 64, "none")
    seg = UNetSegmenter.load(p, "cpu")
    out = seg(np.random.default_rng(0).integers(0, 255, (256, 256)).astype(np.uint8))
    assert out.shape == (256, 256)
