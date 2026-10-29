# Classical image processing + ML pipeline for chest X-rays

Pipeline: preprocess -> enhance -> restore -> segment lungs -> features -> classify -> Streamlit demo.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q tests

## Data
Download the COVID-19 Radiography Database (Kaggle) and unzip to
`data/raw/COVID-19_Radiography_Dataset/` (folders: Normal, Viral Pneumonia, Lung_Opacity, ... each with images/ and masks/).

## Part 1
    python scripts/run_eda.py
    python scripts/run_preprocessing.py --methods clahe homomorphic highpass histspec

    python scripts/run_enhancement_comparison.py --n 30   # tables -> reports/
    jupyter notebook notebooks/02_enhancement_comparison.ipynb   # figures + decision table

## Part 2 (restoration, leakage check, evaluation framework)
Run in this order, before anything is trained:

    python scripts/run_eda.py                         # builds splits.csv
    python scripts/check_duplicates.py                # look at reports/figures/duplicates_check.png
    python scripts/check_duplicates.py --fix          # only if the pairs shown are real duplicates
    python scripts/run_restoration_experiment.py --n 20   # simulated blur/noise: tune K in config.yaml
    python scripts/run_restoration_apply.py           # audit real images, deblur the flagged tail
    python scripts/run_preprocessing.py --restore --methods clahe homomorphic highpass

Restoration is evaluated on *simulated* degradations (real X-rays are rarely blurred) and applied to real
images only where the audit flags them. Every later experiment logs one row to `reports/ablation_table.csv`
via `src/evaluation/ablation.py`.

## Part 3 (lung segmentation)
    python scripts/run_segmentation.py --n 150            # Dice/IoU table on the validation split
    python scripts/run_segmentation.py --write clahe kmeans   # export masks for all images (-> Part 4)

Run `python scripts/build_atlas.py` first (mean lung mask of the train split -> `models/lung_atlas.npy`; also prints
the atlas-only Dice that every real method has to beat).
Methods: otsu, multi_otsu, adaptive, kmeans, region_growing, each plain and with the atlas prior (`+prior`).
Thresholds/clusters are computed on body pixels only; lungs are chosen by anatomy (one per side, mid-image),
not by image-border tricks, because lungs touch the border in tightly cropped X-rays.

### Optional U-Net (needs PyTorch)
    python scripts/train_unet.py --smoke                       # ~1 min check
    python scripts/train_unet.py --enh none --n-train 4000 --epochs 8   # prints a time estimate after 10 steps
    python scripts/run_segmentation.py --unet models/unet_none.pt --enh none
    python scripts/run_segmentation.py --write none unet       # export masks for all images

### Training the U-Net on Google Colab (free GPU)
    python scripts/export_unet_data.py --enh none --size 128   # -> data/colab_bundle/{unet_none_128.npz, code.zip}
Upload both to Drive, then in Colab:  `python scripts/train_unet.py --data unet_none_128.npz --epochs 20 --batch 32`
and copy `models/unet_none.pt` back into this project's `models/` folder.

## Part 4 (feature extraction)
    python scripts/run_segmentation.py --write none unet                       # masks for ALL images
    python scripts/run_feature_extraction.py --enh none --masks none_unet      # -> data/features/features_none__none_unet.csv
    python scripts/analyze_features.py --file data/features/features_none__none_unet.csv
Ablation variants: `--masks gt` (oracle masks), `--masks full` (no segmentation), `--enh clahe|he|histspec|...`.
375 features per image: shape_* (27), stat_* (10), glcm_* (14), hog_* (324). `src.features.extract_features(img, mask)` is what the dashboard calls.
