# Predicting Surface Protein Abundance from RNA Expression in CITE-seq Data

A bioinformatics mini-project based on the [Open Problems - Multimodal Single-Cell Integration](https://www.kaggle.com/competitions/open-problems-multimodal/) Kaggle competition.

## Overview

Predicts surface protein levels (140 targets) from RNA expression profiles using Highly Variable Gene (HVG) selection, TruncatedSVD dimensionality reduction, and Ridge regression on single-cell CITE-seq data from CD34+ hematopoietic stem and progenitor cells (70,988 cells, 22,050 genes, 4 donors).

## Setup

```bash
conda env create -f environment.yml
conda activate cite_project
```

## Data

Download from the [Kaggle competition page](https://www.kaggle.com/competitions/open-problems-multimodal/data). Place all files in the project root directory.

Required files:
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

**Evaluate models (variance-based gene selection):**
```bash
python ridge_svd.py           # Ridge + SVD (70,988 cells)
python PCA.py                 # Ridge + PCA (70,988 cells)
python PLSregression.py       # PLS regression (20,000 cell subsample)
python kernel_ridge_fixed.py  # Kernel Ridge (20,000 cell subsample)
```

**Feature engineering experiments (HVG-based):**
```bash
python Ridge_svd_hvg.py            # Ridge + SVD + HVG (70,988 cells)
python Ridge_svd_hvg_celltype.py   # HVG + cell type encoding
python Ridge_svd_hvg_day.py        # HVG + day feature
python temporal_holdout.py         # Temporal holdout evaluation (days 2-3 → day 4)
python temporal_holdout_day.py     # Temporal holdout + day feature
```


## Data Quality Notes

- 16,397 public test cells attributed to donor 27,678 day 2 were identified as duplicates of donor 32,606 training cells
- Corrupted cells excluded from training, retained in test set for submission integrity
- Corrected donor 27,678 data (7,016 cells) loaded from separate HDF5 file and appended to test set
- Per competition guidelines, predictions for corrupted cells do not affect evaluation score

## Reproducibility

- Python 3.11.11 (conda-forge)
- Random seed fixed to 42 throughout
- Developed on Apple M2 (8GB RAM), macOS