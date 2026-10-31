import sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import make_code_zip

ROOT = Path(__file__).resolve().parents[1]


def test_code_zip_has_forward_slashes_and_key_files(tmp_path):
    out = tmp_path / "code.zip"
    make_code_zip(ROOT, out)
    names = zipfile.ZipFile(out).namelist()
    assert not any("\\" in n for n in names)
    for must in ("config.yaml", "scripts/train_cnn.py", "src/segmentation/unet.py", "tests/phantom.py"):
        assert must in names
