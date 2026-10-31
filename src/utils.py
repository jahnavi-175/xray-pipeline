import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def make_code_zip(root, out_path):
    """Zip src/, scripts/, tests/ (*.py) + config.yaml with forward-slash paths (Colab-friendly)."""
    import zipfile
    from pathlib import Path
    root = Path(root)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for d in ("src", "scripts", "tests"):
            for p in sorted((root / d).rglob("*.py")):
                z.write(p, p.relative_to(root).as_posix())
        z.write(root / "config.yaml", "config.yaml")
