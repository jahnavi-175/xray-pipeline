"""Part 5 (optional): ResNet18 transfer-learning baseline (deep learning vs classical features).
Trains on TRAIN, picks the best epoch on VAL (by AUC), reports VAL (+TEST with --final).
  python scripts/train_cnn.py --smoke                                  # 1-minute check (no downloads)
  python scripts/train_cnn.py --data cnn_none_128.npz --epochs 10 --final
Writes reports/cnn_<enh>_<size>.json (copy it back to your laptop) and models/cnn_<enh>.pt"""
import argparse, copy, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from src.io_utils import ROOT
from src.classifiers.cnn import Net
from src.utils import set_seed
from src.evaluation.metrics import metrics_row
from src.segmentation.unet_data import augment


class DS(Dataset):
    def __init__(self, X, y, train=False, seed=0):
        self.X, self.y, self.train, self.rng = X, y, train, np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.train:
            x, _ = augment(x, np.zeros_like(x), self.rng)
        return torch.from_numpy(x.astype(np.float32) / 255.0)[None], int(self.y[i])


@torch.no_grad()
def predict(model, X, dev, batch=128):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.float32) / 255.0)[:, None].to(dev)
        out.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def evaluate(prefix, model, X, y, cls, dev):
    p = predict(model, X, dev)
    pred = (p > 0.5).astype(int)
    m = metrics_row(prefix, y, pred, p)
    for c in np.unique(cls):
        sel = cls == c
        m[f"{prefix}hit_{c.replace(' ', '_')}"] = float((pred[sel] == y[sel]).mean())
    return m


ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--no-pretrained", action="store_true")
ap.add_argument("--final", action="store_true", help="also report TEST metrics (once, at the very end)")
ap.add_argument("--eval-only", default=None, metavar="CKPT",
                help="evaluate an already-trained checkpoint on VAL and TEST (no training); use once, at the very end")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

set_seed(42)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, "| torch", torch.__version__)
if args.smoke:
    from tests.phantom import make_phantom
    import cv2
    ph = [make_phantom(i, variant=["centered", "wide", "touch", "tight"][i % 4], noise=3 + 8 * (i % 2)) for i in range(48)]
    X = np.stack([cv2.resize(p[0], (64, 64), interpolation=cv2.INTER_AREA) for p in ph])
    y = np.array([i % 2 for i in range(48)]); cls = np.where(y == 0, "Normal", "Abnormal")
    split = np.array(["train"] * 32 + ["val"] * 8 + ["test"] * 8)
    enh, size, args.epochs, args.batch, args.no_pretrained = "smoke", 64, 2, 16, True
else:
    d = np.load(args.data)
    X, y, cls, split = d["X"], d["y"], d["cls"], d["split"]
    enh, size = str(d["enh"]), int(d["size"])
tr, va, te = (split == s for s in ("train", "val", "test"))
print(f"train {tr.sum()} | val {va.sum()} | test {te.sum()} | {size}px | positives {y[tr].mean():.2f}")

rep = ROOT / "reports"; rep.mkdir(exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
js = rep / f"cnn_{enh}_{size}.json"
if args.eval_only:
    ck = torch.load(args.eval_only, map_location="cpu")
    model = Net(pretrained=False).to(dev)
    model.load_state_dict(ck["state"])
    m = {"n_train": int(tr.sum()), **evaluate("val_", model, X[va], y[va], cls[va], dev),
         **evaluate("test_", model, X[te], y[te], cls[te], dev)}
    rec = json.loads(js.read_text()) if js.exists() else {"enh": enh, "size": size, "arch": "resnet18", "pretrained": True, "epochs": None, "metrics": {}}
    rec["metrics"].update(m)
    js.write_text(json.dumps(rec, indent=1))
    print("VAL  acc {:.3f}  F1 {:.3f}  AUC {:.3f}".format(m["val_acc"], m["val_f1"], m["val_auc"]))
    print("TEST acc {:.3f} [{:.3f}, {:.3f}]  F1 {:.3f}  AUC {:.3f}  recall {:.3f}  specificity {:.3f}".format(
        m["test_acc"], m["test_acc_lo"], m["test_acc_hi"], m["test_f1"], m["test_auc"], m["test_rec"], m["test_spec"]))
    print("updated", js)
    sys.exit(0)

model = Net(pretrained=not args.no_pretrained).to(dev)
cnt = np.bincount(y[tr], minlength=2)
weights = torch.tensor(len(y[tr]) / (2.0 * cnt), dtype=torch.float32, device=dev)     # class balancing
crit = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
loader = DataLoader(DS(X[tr], y[tr], train=True), batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=bool(tr.sum() >= args.batch))

best, best_state, best_ep, t0 = -1.0, None, 0, time.time()
for ep in range(1, args.epochs + 1):
    model.train(); run = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step(); run += loss.item()
    sched.step()
    pv = predict(model, X[va], dev)
    auc, acc = roc_auc_score(y[va], pv), float(((pv > .5) == y[va]).mean())
    print(f"epoch {ep}/{args.epochs}  loss {run / len(loader):.4f}  val AUC {auc:.4f}  val acc {acc:.4f}  ({(time.time() - t0) / 60:.1f} min)")
    if auc > best:
        best, best_state, best_ep = auc, copy.deepcopy(model.state_dict()), ep
model.load_state_dict(best_state)
metrics = {"n_train": int(tr.sum()), "best_epoch": best_ep, **evaluate("val_", model, X[va], y[va], cls[va], dev)}
if args.final:
    metrics.update(evaluate("test_", model, X[te], y[te], cls[te], dev))
print(f"best epoch {best_ep}: val acc {metrics['val_acc']:.3f}  F1 {metrics['val_f1']:.3f}  AUC {metrics['val_auc']:.3f}")

js.write_text(json.dumps({"enh": enh, "size": size, "arch": "resnet18", "pretrained": not args.no_pretrained,
                          "epochs": args.epochs, "metrics": metrics}, indent=1))
torch.save({"state": model.state_dict(), "enh": enh, "size": size, "arch": "resnet18"}, ROOT / "models" / f"cnn_{enh}.pt")
print("wrote", js, "and models/", f"cnn_{enh}.pt")
