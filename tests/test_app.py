import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import joblib
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from tests.synth_world import build_world

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    bundle, atlas, imgs, gts = build_world(40)
    models, samples = tmp_path / "models", tmp_path / "samples"
    models.mkdir(); samples.mkdir()
    joblib.dump(bundle, models / "classifier.joblib"); np.save(models / "lung_atlas.npy", atlas)
    cv2.imwrite(str(samples / "sample_a.png"), cv2.resize(imgs[1], (299, 299)))
    monkeypatch.setenv("XRAY_MODELS_DIR", str(models)); monkeypatch.setenv("XRAY_SAMPLES_DIR", str(samples))
    import streamlit as st
    st.cache_resource.clear(); st.cache_data.clear()
    return AppTest.from_file(str(APP), default_timeout=120)


def test_starts_without_image_and_without_crashing(app):
    at = app.run()
    assert not at.exception
    assert any("Upload an image" in i.value for i in at.info)


def test_full_flow_on_sample_and_threshold_changes_verdict(app):
    at = app.run()
    at.selectbox(key="sample").select("sample_a.png").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Probability of abnormal" in [m.label for m in at.metric]
    assert any("U-Net unavailable" in w.value for w in at.warning)          # graceful fallback
    at.slider(key="thr").set_value(0.05).run()
    assert at.error and not at.success                                        # everything "abnormal" at 5%
    at.slider(key="thr").set_value(0.95).run()
    assert at.success and not at.error


def test_other_segmentation_and_enhancement_options_work(app):
    at = app.run()
    at.selectbox(key="sample").select("sample_a.png").run()
    at.selectbox(key="seg").select("otsu+prior").run()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key="enh_view").select("clahe").run()
    assert not at.exception, [e.value for e in at.exception]
    at.checkbox(key="restore").uncheck().run()
    assert not at.exception, [e.value for e in at.exception]


def test_cnn_second_opinion_shown_and_agreement_reported(app, monkeypatch):
    from types import SimpleNamespace
    import src.pipeline as pl
    real = pl.load_models

    def with_cnn(*a, **k):
        m = real(*a, **k)
        m.cnn = SimpleNamespace(enh="none", predict_prob=lambda im: 0.9)
        return m

    monkeypatch.setattr(pl, "load_models", with_cnn)
    at = app.run()
    at.selectbox(key="sample").select("sample_a.png").run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert any("CNN (ResNet18)" in l for l in labels) and any("Feature-based" in l for l in labels)
    at.slider(key="thr").set_value(0.95).run()                                 # CNN 0.9 < 0.95 -> same side as low-prob model
    assert not at.exception
