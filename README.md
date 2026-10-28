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
