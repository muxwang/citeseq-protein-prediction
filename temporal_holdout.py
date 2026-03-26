import numpy as np
import pandas as pd
import tables as tb
import scanpy as sc
import anndata as ad
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import time

start_time = time.time()


# Settings

USE_TOP_GENES = 2000
N_COMPONENTS = 100
ALPHA = 1.0
RANDOM_SEED = 42


# Helper functions

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



# 1. Load cell IDs from HDF5

print("Loading cell IDs...")
with tb.File("train_cite_inputs.h5", "r") as f:
    cell_ids_bytes = f.root.train_cite_inputs.axis1[:]
    gene_ids_bytes = f.root.train_cite_inputs.axis0[:]
cell_ids = [cid.decode("utf-8") for cid in cell_ids_bytes]
gene_ids = [gid.decode("utf-8") for gid in gene_ids_bytes]
print(f"Total cells: {len(cell_ids)}")


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


# 3. Load full RNA and protein data

print("\nLoading full RNA data...")
with tb.File("train_cite_inputs.h5", "r") as f:
    all_idx = list(range(len(cell_ids)))
    X = f.root.train_cite_inputs.block0_values[:]

print("Loading full protein data...")
with tb.File("train_cite_targets.h5", "r") as f:
    y = f.root.train_cite_targets.block0_values[:]

# Apply clean mask from corrupted donor removal
clean_indices = np.where(clean_mask)[0]
X = X[clean_indices]
y = y[clean_indices]

X = X.astype(np.float32)
y = y.astype(np.float32)
print(f"Full X shape: {X.shape}")
print(f"Full y shape: {y.shape}")


# 4. Temporal split — train days 2-3, validate day 4
#    Day 4 is used as proxy for day 7 (true test condition)

print("\nSplitting by time point...")
train_mask = (days == 2) | (days == 3)
val_mask = days == 4

X_train = X[train_mask]
y_train = y[train_mask]
X_val = X[val_mask]
y_val = y[val_mask]

print(f"Train cells (days 2-3): {X_train.shape[0]}")
print(f"Val cells   (day 4):    {X_val.shape[0]}")
print(f"Train donors: {np.unique(donors[train_mask])}")
print(f"Val donors:   {np.unique(donors[val_mask])}")


# 5. HVG selection on training days only

print(f"\nSelecting top {USE_TOP_GENES} HVGs on training data...")
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


# 6. SVD on training days only

print(f"Fitting TruncatedSVD with {N_COMPONENTS} components...")
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
X_train_svd = svd.fit_transform(X_train_sel)
X_val_svd = svd.transform(X_val_sel)


# 7. Ridge on training days only

print(f"Fitting Ridge(alpha={ALPHA})...")
ridge = Ridge(alpha=ALPHA)
ridge.fit(X_train_svd, y_train)
y_pred = ridge.predict(X_val_svd)


# 8. Evaluate

print("\nEvaluating...")
model_score = compute_score(y_val, y_pred)
dummy_score = run_dummy_baseline(y_train, y_val)

print("\n" + "=" * 60)
print("TEMPORAL HOLDOUT EVALUATION")
print("Train: days 2-3 | Validate: day 4 (proxy for day 7)")
print("=" * 60)
print(f"Model score:   {model_score:.6f}")
print(f"Dummy score:   {dummy_score:.6f}")
print(f"Gain over dummy: {model_score - dummy_score:.6f}")
print(f"\nFor comparison:")
print(f"GroupKFold CV score (days 2-4): 0.882085")
print(f"Temporal holdout score (day 4): {model_score:.6f}")
print(f"Optimism in CV estimate: {0.882085 - model_score:.6f}")
print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")