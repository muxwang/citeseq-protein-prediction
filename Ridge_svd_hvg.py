import numpy as np
import pandas as pd
import tables as tb
import scanpy as sc
import anndata as ad
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
import time

start_time = time.time()


# Settings — identical to Ridge_svd.py except gene selection

SAMPLE_SIZE = 70988
USE_TOP_GENES = 2000
N_COMPONENTS = 100
ALPHA = 1.0
N_SPLITS = 5
RANDOM_SEED = 42


# Helper functions — identical to other scripts

def compute_score(y_true, y_pred):
    corrs = []
    for i in range(y_true.shape[0]):
        true_row = np.asarray(y_true[i]).ravel()
        pred_row = np.asarray(y_pred[i]).ravel()
        if np.std(true_row) == 0 or np.std(pred_row) == 0:
            corr = -1.0
        else:
            corr, _ = pearsonr(true_row, pred_row)
            if np.isnan(corr):
                corr = -1.0
        corrs.append(corr)
    return float(np.mean(corrs))


def run_dummy_baseline(y_train, y_val):
    y_dummy = np.tile(y_train.mean(axis=0), (y_val.shape[0], 1))
    return compute_score(y_val, y_dummy)


def run_permutation_test(X_train_svd, y_train, X_val_svd, y_val, random_seed):
    y_train_perm = y_train.copy()
    rng = np.random.default_rng(random_seed)
    for j in range(y_train_perm.shape[1]):
        rng.shuffle(y_train_perm[:, j])
    ridge_perm = Ridge(alpha=ALPHA)
    ridge_perm.fit(X_train_svd, y_train_perm)
    y_pred_perm = ridge_perm.predict(X_val_svd)
    return compute_score(y_val, y_pred_perm)



# 1. Load cell IDs from HDF5

print("Loading cell IDs...")
with tb.File("train_cite_inputs.h5", "r") as f:
    cell_ids_bytes = f.root.train_cite_inputs.axis1[:]
    gene_ids_bytes = f.root.train_cite_inputs.axis0[:]
cell_ids = [cid.decode("utf-8") for cid in cell_ids_bytes]
gene_ids = [gid.decode("utf-8") for gid in gene_ids_bytes]
n_total = len(cell_ids)
print(f"Total cells: {n_total}")


# 2. Load metadata and align to HDF5 order

print("Loading metadata...")
metadata = pd.read_csv("metadata.csv")
metadata_filtered = metadata[metadata["cell_id"].isin(cell_ids)].copy()
metadata_filtered = metadata_filtered.set_index("cell_id").loc[cell_ids].reset_index()

days = metadata_filtered["day"].to_numpy()
donors = metadata_filtered["donor"].to_numpy()


# 2.5 Remove corrupted donor 27678 cells

print("Removing corrupted donor 27678 cells...")
bad_mask = (metadata_filtered["donor"] == 27678) & (metadata_filtered["day"] == 2)
clean_mask = ~bad_mask
metadata_filtered = metadata_filtered[clean_mask].reset_index(drop=True)
cell_ids = metadata_filtered["cell_id"].tolist()
days = metadata_filtered["day"].to_numpy()
donors = metadata_filtered["donor"].to_numpy()
print(f"Cells after removing corrupted donor: {len(cell_ids)}")


# 3. Restrict to CITE training days and sample cells

print("\nSampling cells...")
np.random.seed(RANDOM_SEED)

valid_days = [2, 3, 4]
valid_mask = metadata_filtered["day"].isin(valid_days).to_numpy()
valid_idx = np.where(valid_mask)[0]
days_valid = days[valid_idx]

sample_size = min(SAMPLE_SIZE, len(valid_idx))
day_indices = {day: valid_idx[days_valid == day] for day in valid_days}

sample_idx = []
for day in valid_days:
    idx_day = day_indices[day]
    n_avail = len(idx_day)
    if n_avail == 0:
        continue
    n_sample_day = max(1, min(int(sample_size * n_avail / len(valid_idx)), n_avail))
    chosen = np.random.choice(idx_day, n_sample_day, replace=False)
    sample_idx.extend(chosen)

sample_idx = np.array(sorted(sample_idx))
days_sampled = days[sample_idx]
donors_sampled = donors[sample_idx]
print(f"Sampled {len(sample_idx)} cells")


# 4. Load RNA and protein data for sampled cells

print("\nLoading RNA subset...")
with tb.File("train_cite_inputs.h5", "r") as f:
    X = f.root.train_cite_inputs.block0_values[sample_idx, :]

print("Loading protein subset...")
with tb.File("train_cite_targets.h5", "r") as f:
    y = f.root.train_cite_targets.block0_values[sample_idx, :]

X = X.astype(np.float32)
y = y.astype(np.float32)
print(f"Loaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")


# 5. Build grouped CV labels

groups = np.array([f"{donor}_{day}" for donor, day in zip(donors_sampled, days_sampled)])
unique_groups = np.unique(groups)
print(f"\nUnique donor-day groups: {len(unique_groups)}")

if len(unique_groups) < N_SPLITS:
    raise ValueError(f"Only {len(unique_groups)} groups but N_SPLITS={N_SPLITS}")

gkf = GroupKFold(n_splits=N_SPLITS)


# 6. Cross-validation loop

fold_scores = []
fold_dummy_scores = []
fold_perm_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    print("\n" + "=" * 60)
    print(f"Fold {fold}/{N_SPLITS}")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    train_groups = groups[train_idx]
    val_groups = groups[val_idx]
    print(f"Train cells: {X_train.shape[0]}, Val cells: {X_val.shape[0]}")
    print(f"Train groups: {np.unique(train_groups)}")
    print(f"Val groups:   {np.unique(val_groups)}")

    
    # HVG selection on TRAIN fold only using scanpy
    # Uses cell_ranger flavor which works on log-normalized data
    # Unlike variance selection, HVG accounts for mean-variance
    # relationship to identify genuinely variable genes
    
    print(f"Selecting top {USE_TOP_GENES} HVGs using scanpy...")
    adata_train = ad.AnnData(
        X=X_train,
        obs=pd.DataFrame(index=range(X_train.shape[0])),
        var=pd.DataFrame(index=gene_ids)
    )
    sc.pp.highly_variable_genes(
        adata_train,
        n_top_genes=USE_TOP_GENES,
        flavor="cell_ranger"
    )
    hvg_mask = adata_train.var["highly_variable"].values
    top_idx = np.where(hvg_mask)[0]

    X_train_sel = X_train[:, top_idx]
    X_val_sel = X_val[:, top_idx]
    print(f"Selected {X_train_sel.shape[1]} HVGs")

    
    # SVD on train fold only
    
    n_comp_fold = min(N_COMPONENTS, X_train_sel.shape[0] - 1, X_train_sel.shape[1])
    svd = TruncatedSVD(n_components=n_comp_fold, random_state=RANDOM_SEED)
    X_train_svd = svd.fit_transform(X_train_sel)
    X_val_svd = svd.transform(X_val_sel)

    
    # Ridge on train fold only
    
    ridge = Ridge(alpha=ALPHA)
    ridge.fit(X_train_svd, y_train)
    y_pred = ridge.predict(X_val_svd)

    fold_score = compute_score(y_val, y_pred)
    fold_scores.append(fold_score)
    print(f"Fold {fold} model score: {fold_score:.6f}")

    dummy_score = run_dummy_baseline(y_train, y_val)
    fold_dummy_scores.append(dummy_score)
    print(f"Fold {fold} dummy score: {dummy_score:.6f}")

    perm_score = run_permutation_test(
        X_train_svd, y_train, X_val_svd, y_val,
        random_seed=RANDOM_SEED + fold
    )
    fold_perm_scores.append(perm_score)
    print(f"Fold {fold} permutation score: {perm_score:.6f}")


# 7. Final summary

print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY — Ridge + SVD + HVG Selection")
print("=" * 60)

for i in range(N_SPLITS):
    print(f"Fold {i+1}: "
          f"model={fold_scores[i]:.6f}, "
          f"dummy={fold_dummy_scores[i]:.6f}, "
          f"perm={fold_perm_scores[i]:.6f}")

print("\nAverage model score:       {:.6f}".format(np.mean(fold_scores)))
print("Average dummy score:       {:.6f}".format(np.mean(fold_dummy_scores)))
print("Average permutation score: {:.6f}".format(np.mean(fold_perm_scores)))
print("\nStd model score:           {:.6f}".format(np.std(fold_scores)))
print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")