"""Part 3 (optional): train a small U-Net on the dataset's lung masks (TRAIN split only; the VAL split
picks the best epoch; TEST is never touched here).

  python scripts/train_unet.py --smoke                     # 1-minute check that everything works
  python scripts/train_unet.py --enh none --n-train 4000 --epochs 8
  python scripts/train_unet.py --data unet_none_128.npz --epochs 20 --batch 32      # Colab (see export_unet_data.py)
Then evaluate:  python scripts/run_segmentation.py --unet models/unet_none.pt --enh none"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.io_utils import ROOT, load_config
from src.utils import set_seed
from src.segmentation.unet import (UNet, XrayDataset, dice_loss, predict_probs, save_checkpoint, UNetSegmenter)
from src.segmentation.unet_data import load_arrays
from src.evaluation.metrics import dice

ap = argparse.ArgumentParser()
ap.add_argument("--enh", default="none", help="which image variant to train on (none = processed)")
ap.add_argument("--size", type=int, default=128, help="training resolution (128 = fast on CPU)")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--n-train", type=int, default=4000, help="class-balanced training subset size")
ap.add_argument("--n-val", type=int, default=300)
ap.add_argument("--base", type=int, default=16, help="channels in the first layer")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--device", default="auto")
ap.add_argument("--data", default=None, help="npz made by export_unet_data.py (use this on Colab)")
ap.add_argument("--smoke", action="store_true", help="tiny synthetic run to verify the setup")
args = ap.parse_args()

cfg = load_config(); set_seed(cfg["seed"])
dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                   ("cpu" if args.device == "auto" else args.device))
print("device:", dev, "| torch", torch.__version__)

if args.smoke:
    from tests.phantom import make_phantom
    import cv2
    pairs = [make_phantom(i, variant=v) for i, v in enumerate(["centered", "wide", "touch", "tight"] * 8)]
    Xs = np.stack([cv2.resize(p[0], (args.size, args.size), interpolation=cv2.INTER_AREA) for p in pairs])
    Ys = np.stack([(cv2.resize(p[1].astype(np.float32), (args.size, args.size), interpolation=cv2.INTER_AREA) > .5).astype(np.uint8) for p in pairs])
    Xtr, Ytr, Xva, Yva = Xs[:24], Ys[:24], Xs[24:], Ys[24:]
    args.epochs, args.batch, out_path = 3, 8, ROOT / "models" / "unet_smoke.pt"
elif args.data:
    d = np.load(args.data)
    Xtr, Ytr, Xva, Yva = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"]
    args.enh, args.size = str(d["enh"]), int(d["size"])
    out_path = ROOT / "models" / f"unet_{args.enh}.pt"
else:
    df = pd.read_csv(ROOT / cfg["paths"]["features"] / "splits.csv")
    Xtr, Ytr = load_arrays(df[df.split == "train"], cfg, args.enh, args.size, args.n_train, cfg["seed"])
    Xva, Yva = load_arrays(df[df.split == "val"], cfg, args.enh, args.size, args.n_val, cfg["seed"])
    out_path = ROOT / "models" / f"unet_{args.enh}.pt"
print(f"train {len(Xtr)} | val {len(Xva)} | {args.size}px | lung fraction {Ytr.mean():.3f}")

out_path.parent.mkdir(parents=True, exist_ok=True)
model = UNet(args.base).to(dev)
print("parameters: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
loader = DataLoader(XrayDataset(Xtr, Ytr, train=True, seed=cfg["seed"]), batch_size=args.batch,
                    shuffle=True, num_workers=0, drop_last=len(Xtr) >= args.batch)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
bce = torch.nn.BCEWithLogitsLoss()

best, t0, step = -1.0, time.time(), 0
total_steps = args.epochs * len(loader)
for ep in range(1, args.epochs + 1):
    model.train(); run = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        logits = model(xb)
        loss = bce(logits, yb) + dice_loss(logits, yb)
        loss.backward(); opt.step()
        run += loss.item(); step += 1
        if step == 10:
            eta = (time.time() - t0) / step * total_steps / 60
            print(f"  ~{eta:.1f} min total estimated (from the first 10 steps)")
    sched.step()
    P = predict_probs(model, Xva, dev) > 0.5
    vd = float(np.mean([dice(p, g) for p, g in zip(P, Yva)]))
    print(f"epoch {ep}/{args.epochs}  loss {run / len(loader):.4f}  val Dice {vd:.4f}  ({(time.time() - t0) / 60:.1f} min)")
    if vd > best:
        best = vd
        save_checkpoint(out_path, model, args.base, args.size, args.enh if not args.smoke else "none", {"val_dice": vd})
print(f"best val Dice {best:.4f} -> {out_path}")

seg = UNetSegmenter.load(out_path, dev)                      # round-trip check of the saved file
test_img = (Xva[0] if args.smoke else Xva[0])
print("reload OK, mask pixels:", int(seg(test_img).sum()))
