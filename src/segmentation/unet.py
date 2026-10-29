"""Small U-Net for lung segmentation (needs PyTorch). Trained by scripts/train_unet.py."""
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_fill_holes
from .postprocess import keep_largest
from .unet_data import augment


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """4 down / 4 up levels. Input (B,1,H,W) with H, W divisible by 16. Output = logits (B,1,H,W)."""

    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([DoubleConv(1, c[0]), DoubleConv(c[0], c[1]),
                                  DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3])])
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(c[3], c[4])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[4], c[3], 2, 2), nn.ConvTranspose2d(c[3], c[2], 2, 2),
                                 nn.ConvTranspose2d(c[2], c[1], 2, 2), nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([DoubleConv(c[3] * 2, c[3]), DoubleConv(c[2] * 2, c[2]),
                                  DoubleConv(c[1] * 2, c[1]), DoubleConv(c[0] * 2, c[0])])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bott(x)
        for up, dec, sk in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([sk, up(x)], dim=1))
        return self.out(x)


class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, train=False, seed=0):
        self.X, self.Y, self.train = X, Y, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.train:
            x, y = augment(x, y, self.rng)
        return (torch.from_numpy(x.astype(np.float32) / 255.0)[None],
                torch.from_numpy(y.astype(np.float32))[None])


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def predict_probs(model, X, device, batch=32):
    """X: (N,H,W) uint8 -> (N,H,W) float probabilities."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32) / 255.0)[:, None].to(device)
        out.append(torch.sigmoid(model(xb))[:, 0].cpu().numpy())
    return np.concatenate(out)


def save_checkpoint(path, model, base, size, enh, extra=None):
    torch.save({"state": model.state_dict(), "base": base, "size": size, "enh": enh, **(extra or {})}, path)


class UNetSegmenter:
    """Inference wrapper: uint8 grayscale image (any size) -> uint8 {0,1} mask. Used by the dashboard."""

    def __init__(self, model, size, enh, device="cpu"):
        self.model, self.size, self.enh, self.device = model.to(device).eval(), size, enh, device

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location="cpu")
        model = UNet(ck["base"])
        model.load_state_dict(ck["state"])
        return cls(model, ck["size"], ck["enh"], device)

    @torch.no_grad()
    def predict_prob(self, img):
        x = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = torch.from_numpy(x)[None, None].to(self.device)
        p = torch.sigmoid(self.model(t))[0, 0].cpu().numpy()
        return cv2.resize(p, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    def __call__(self, img, thresh=0.5):
        m = (self.predict_prob(img) > thresh)
        m = binary_fill_holes(m).astype(np.uint8)
        return keep_largest(m, n=2, min_area_frac=0.005)
