"""Config loading, dataset indexing, image I/O."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="config.yaml"):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def load_gray(path):
    """Read any image as uint8 grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_img(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def build_index(cfg):
    """Scan the COVID-19 Radiography layout: <root>/<Class>/images/*.png (+ masks/*.png).
    Returns a DataFrame with image_id, class_name, label, path, mask_path, split."""
    root = ROOT / cfg["paths"]["raw_root"]
    rows = []
    for cls_name, label in cfg["classes"].items():
        img_dir = root / cls_name / "images"
        for p in sorted(img_dir.glob("*.png")):
            m = root / cls_name / "masks" / p.name
            rows.append(dict(image_id=p.stem, class_name=cls_name, label=label,
                             path=str(p), mask_path=str(m) if m.exists() else None))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {root}. Check config.yaml paths/classes.")
    seed = cfg["seed"]
    test, val = cfg["split"]["test"], cfg["split"]["val"]
    trainval, te = train_test_split(df, test_size=test, stratify=df["label"], random_state=seed)
    tr, va = train_test_split(trainval, test_size=val / (1 - test),
                              stratify=trainval["label"], random_state=seed)
    df["split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[te.index, "split"] = "test"
    return df


def load_mask(path, size=256):
    """Binary lung mask (0/1, uint8) resized with nearest-neighbour to match the images."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def enhanced_path(cfg, enh, image_id):
    """Path of an image for a given enhancement variant ('none' = processed, un-enhanced)."""
    base = ROOT / cfg["paths"]["processed"] if enh == "none" else ROOT / cfg["paths"]["enhanced"] / enh
    return base / f"{image_id}.png"
