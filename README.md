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

## Part 2 (restoration)
    python scripts/run_restoration_experiment.py --n 20   # PSNR/SSIM table + example figure

Restoration is evaluated on *simulated* degradations (real X-rays are rarely blurred);
use `src/restoration.py` on real data only if the noise/blur estimate says it is needed.
