"""Chest X-ray analysis dashboard.   Run:  streamlit run app/streamlit_app.py
upload -> preprocessing -> enhancement -> segmentation -> features -> prediction"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from src.io_utils import load_config
from src.enhancement import quality_report
from src.pipeline import (load_models, analyze, segment_with, overlay_mask, available_enhancements, operating_points,
                          threshold_for_sensitivity, reference_table, sample_dir)

st.set_page_config(page_title="Chest X-ray pipeline", layout="wide")
CFG = load_config()


@st.cache_resource(show_spinner="Loading models...")
def get_models():
    return load_models(os.environ.get("XRAY_MODELS_DIR"), CFG)


@st.cache_data(show_spinner="Analysing image...", max_entries=20)
def run(img_bytes, seg, enh_view, restore, _models):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return analyze(img, _models, CFG, seg=seg, enh_view=enh_view, restore=restore)


def hist_fig(images: dict):
    fig, ax = plt.subplots(1, len(images), figsize=(4.2 * len(images), 2.4))
    ax = np.atleast_1d(ax)
    for a, (name, im) in zip(ax, images.items()):
        a.hist(im.ravel(), bins=64, range=(0, 255), color="steelblue"); a.set_title(name, fontsize=10)
        a.set_yticks([]); a.set_xlim(0, 255)
    fig.tight_layout()
    return fig


models = get_models()
clf = models.clf

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
seg_opts = ["unet"] + [m + ("+prior" if models.atlas is not None else "")
                       for m in ("adaptive", "otsu", "kmeans", "multi_otsu", "region_growing")]
seg = st.sidebar.selectbox("Segmentation method", seg_opts, key="seg",
                           help="U-Net is the accurate one (Dice ~0.98). The classical methods are baselines (Dice 0.4-0.75).")
enh_view = st.sidebar.selectbox("Enhancement to preview", available_enhancements(models), key="enh_view",
                                help="Display only. Features always use the enhancement the classifier was trained on.")
restore = st.sidebar.checkbox("Deblur (Wiener) if the image looks blurry", value=True, key="restore")
threshold = st.sidebar.slider("Decision threshold (abnormal if probability >=)", 0.05, 0.95, 0.50, 0.01, key="thr",
                              help="Lower = catches more abnormal cases but raises false alarms.")
for n in models.notes:
    st.sidebar.warning(n)
st.sidebar.caption("Research prototype for a coursework project. Not a medical device; not clinically validated.")

# ------------------------------------------------------------------ input
st.title("Chest X-ray analysis pipeline")
st.write("Upload a frontal chest X-ray (PNG/JPG). The app enhances it, segments the lungs, measures shape and texture "
         "features and predicts **Normal vs Abnormal** (lung opacity / viral pneumonia).")
up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="upload")
samples = sorted(p for p in sample_dir().glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
choice = st.selectbox("...or use a sample image", ["(none)"] + [p.name for p in samples], key="sample") if samples else "(none)"

img_bytes, label = None, ""
if up is not None:
    img_bytes, label = up.getvalue(), up.name
elif choice != "(none)":
    img_bytes, label = (sample_dir() / choice).read_bytes(), choice
if img_bytes is None:
    st.info("Upload an image or pick a sample to start.")
    st.stop()

r = run(img_bytes, seg, enh_view, restore, models)
if r is None:
    st.error("This file could not be read as an image.")
    st.stop()

prob, cprob = r["prob"], r["cnn_prob"]
pred = None if prob is None else int(prob >= threshold)
cpred = None if cprob is None else int(cprob >= threshold)

# ------------------------------------------------------------------ summary
c1, c2, c3 = st.columns([1, 1, 1.4])
c1.image(r["original"], caption=f"Input: {label}", width=300)
c2.image(r["overlay"], caption=f"Lung segmentation ({r['seg_used']})", width=300)
with c3:
    if prob is None:
        st.metric("Probability of abnormal", "n/a")
    else:
        st.metric("Probability of abnormal", f"{prob:.0%}")
        st.progress(min(max(prob, 0.0), 1.0))
        (st.error if pred else st.success)(
            f"**{'Abnormal' if pred else 'Normal'}** at threshold {threshold:.2f}")
        if cprob is not None:
            st.caption(f"Second opinion (CNN): {cprob:.0%} abnormal -> "
                       + ("agrees" if cpred == pred else "**disagrees**, look at the image carefully"))
    for w in r["warnings"]:
        st.warning(w)

tabs = st.tabs(["1. Preprocessing", "2. Enhancement", "3. Segmentation", "4. Features", "5. Prediction"])

with tabs[0]:
    a, b = st.columns(2)
    a.image(r["original"], caption="Resized to 256x256", width=300)
    b.image(r["processed"], caption="Denoised" + (" + deblurred" if r["restored"] else ""), width=300)
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated noise sigma", f"{r['noise_sigma']:.2f}")
    m2.metric("Sharpness (Laplacian variance)", f"{r['blur_score']:.0f}")
    m3.metric("Restoration", "applied" if r["restored"] else ("flagged, off" if r["flagged_blurry"] else "not needed"))
    st.caption("Median 3x3 denoising; Wiener deconvolution is applied only to images in the blurriest 5% of the training "
               "set (Unit IV).")

with tabs[1]:
    a, b = st.columns(2)
    a.image(r["processed"], caption="Before", width=300)
    b.image(r["enhanced_view"], caption=f"After: {enh_view}", width=300)
    st.pyplot(hist_fig({"before": r["processed"], f"after ({enh_view})": r["enhanced_view"]}))
    q0, q1 = quality_report(r["processed"]), quality_report(r["enhanced_view"])
    st.dataframe(pd.DataFrame({"before": q0, "after": q1}).round(2))
    st.info(f"Features for the classifier are computed on the **{r['feature_enh']}** image, because that is what it was trained on.")

with tabs[2]:
    a, b = st.columns(2)
    a.image(r["overlay"], caption=f"Method: {r['seg_used']}", width=300)
    b.image((r["mask"] * 255).astype(np.uint8), caption="Binary mask", width=300)
    st.metric("Lung area (% of image)", f"{r['lung_area_pct']:.1f}")
    with st.expander("Compare all segmentation methods on this image"):
        if st.button("Run comparison", key="cmp"):
            names = ["unet"] + [m + ("+prior" if models.atlas is not None else "")
                                for m in ("adaptive", "otsu", "kmeans", "region_growing")]
            cols = st.columns(len(names))
            for col, nm in zip(cols, names):
                mk, used, _ = segment_with(r["processed"], models, nm)
                col.image(overlay_mask(r["processed"], mk), caption=used, width=170)

with tabs[3]:
    f = {k: v for k, v in r["features"].items() if k != "mask_ok"}
    groups = {g: {k: v for k, v in f.items() if k.startswith(g + "_")} for g in ("shape", "stat", "glcm", "hog")}
    g1, g2, g3, g4 = st.columns(4)
    for col, (g, d) in zip((g1, g2, g3, g4), groups.items()):
        col.metric({"shape": "Shape features", "stat": "Intensity stats", "glcm": "GLCM texture", "hog": "HOG values"}[g], len(d))
    for g, title in (("shape", "Shape (from the lung mask)"), ("stat", "Intensity statistics inside the lungs"),
                     ("glcm", "GLCM texture inside the lungs")):
        with st.expander(title, expanded=(g == "shape")):
            st.dataframe(pd.DataFrame({"value": groups[g]}).round(4))
    st.caption("324 HOG values (gradient orientations of the lung region) are also computed but not listed here.")
    ref = reference_table(r["features"], clf)
    if ref is not None:
        st.subheader("This image vs typical training images")
        st.dataframe(ref.round(3), hide_index=True)

with tabs[4]:
    if prob is None:
        st.warning("No prediction available (see warnings above).")
    else:
        p1, p2 = st.columns(2)
        p1.metric("Feature-based model: probability of abnormal", f"{prob:.1%}",
                  help="Classifier on shape / intensity / GLCM / HOG features of the segmented lungs (interpretable).")
        if cprob is not None:
            p2.metric("CNN (ResNet18): probability of abnormal", f"{cprob:.1%}",
                      help="End-to-end deep network on the processed image (more accurate in testing, but a black box).")
            if cpred == pred:
                st.success("Both models give the same verdict at this threshold.")
            else:
                st.warning("The two models disagree at this threshold. Treat this image as uncertain.")
        else:
            p2.caption("CNN not available (models/cnn_none.pt missing or PyTorch not installed).")
        ops = operating_points(clf)
        if ops is not None:
            s, y = np.asarray(clf["val_scores"]), np.asarray(clf["val_labels"])
            sens, spec = float((s >= threshold)[y == 1].mean()), float((s < threshold)[y == 0].mean())
            m1, m2 = st.columns(2)
            m1.metric("Sensitivity at this threshold (validation)", f"{sens:.0%}", help="Share of abnormal cases caught.")
            m2.metric("Specificity at this threshold (validation)", f"{spec:.0%}", help="Share of normal cases correctly cleared.")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(ops.threshold, ops.sensitivity, label="sensitivity"); ax.plot(ops.threshold, ops.specificity, label="specificity")
            ax.axvline(threshold, color="k", ls=":"); ax.set_xlabel("threshold"); ax.legend(); fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            tgt = threshold_for_sensitivity(clf, 0.90)
            if tgt:
                st.info(f"To catch about 90% of abnormal cases, use a threshold near {tgt['threshold']:.2f} "
                        f"(specificity then about {tgt['specificity']:.0%}).")
        else:
            st.caption("Re-save the classifier (train_classifier.py --save) to enable the threshold trade-off curve.")
    with st.expander("Model and limitations"):
        if clf:
            st.write(f"Classifier: **{clf['kind'].upper()}** on {len(clf['columns'])} features "
                     f"({', '.join(clf['groups'])}); features from the **{clf['enh']}** image; masks from **{clf['masks']}**.")
        st.write("- The threshold curve and sensitivity/specificity shown are for the feature-based model on the validation set; "
                 "the CNN probabilities are not calibrated the same way.\n"
                 "- Trained on one public dataset (COVID-19 Radiography Database), binary task: Normal vs lung opacity / viral pneumonia.\n"
                 "- Classes come from different source collections and the dataset has no patient IDs, so results may be optimistic.\n"
                 "- Ground-truth lung masks were produced semi-automatically, not by radiologists.\n"
                 "- Not validated clinically. Do not use for diagnosis.")
    with st.expander("Timings"):
        st.json({k: f"{v * 1000:.0f} ms" for k, v in r["timings"].items()})
