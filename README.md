# Predicting Surface Protein Abundance from RNA Expression in CITE-seq Data

A bioinformatics mini-project based on the [Open Problems - Multimodal Single-Cell Integration](https://www.kaggle.com/competitions/open-problems-multimodal/) Kaggle competition.

## Overview

Predicts surface protein levels (140 targets) from RNA expression profiles using variance-based gene selection, TruncatedSVD dimensionality reduction, and Ridge regression on single-cell CITE-seq bone marrow data (70,988 cells, 22,050 genes).

## Requirements

- Python 3.11.7
- See `requirements.txt` for full dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download from the [Kaggle competition page](https://www.kaggle.com/competitions/open-problems-multimodal/data).

Required files — place all in the project root directory:

- `train_cite_inputs.h5`
- `train_cite_targets.h5`
- `test_cite_inputs.h5`
- `test_cite_inputs_day_2_donor_27678.h5`
- `metadata.csv`
- `metadata_cite_day_2_donor_27678.csv`
- `evaluation_ids.csv`

## Usage

**Generate predictions:**
```bash
python final_submission.py
```
Output: `submission_cite_only.csv`

**Evaluate models:**
```bash
python Ridge_svd.py           # Ridge + SVD (full dataset, 70,988 cells)
python PCA.py                 # Ridge + PCA (full dataset, 70,988 cells)
python PLSregression.py       # PLS regression (20,000 cell subsample)
python kernel_ridge_fixed.py  # Kernel Ridge (20,000 cell subsample)
```

## Results

| Model | Cells | Mean Pearson r | Dummy Baseline | Gain |
|---|---|---|---|---|
| Ridge + SVD | 70,988 | 0.8779 | 0.8011 | 0.0768 |
| Ridge + PCA | 70,988 | 0.8780 | 0.8011 | 0.0769 |
| PLS Regression | 20,000 | 0.8750 | 0.8012 | 0.0739 |
| Kernel Ridge | 20,000 | 0.8757 | 0.8012 | 0.0745 |

Evaluated using 5-fold GroupKFold cross-validation grouped by donor and time point.



## Reproducibility

- Python 3.11.7
- Random seed fixed to 42 throughout
- Developed on Apple M2 (8GB RAM), macOS