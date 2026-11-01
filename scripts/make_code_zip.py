"""Rebuild data/colab_bundle/code.zip (src, scripts, tests, config) without re-exporting any data.
Upload it to your Drive folder 'xray' (replace the old one) before the next Colab session."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import ROOT
from src.utils import make_code_zip

out = ROOT / "data" / "colab_bundle"; out.mkdir(parents=True, exist_ok=True)
make_code_zip(ROOT, out / "code.zip")
print("wrote", out / "code.zip")
