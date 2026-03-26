import numpy as np
import pandas as pd
import tables as tb
import scanpy as sc
import anndata as ad
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge


# Settings: final chosen config
# HVG selection chosen over variance for biological motivation
# No day/cell type features — generalizes better to day 10
# and donor 27678 hidden cells

USE_TOP_GENES = 2000
N_COMPONENTS = 100
ALPHA = 1.0
RANDOM_SEED = 42


# 1. Load full training data

print("Loading full training data...")
with tb.File("train_cite_inputs.h5", "r") as f:
    X_train = f.root.train_cite_inputs.block0_values[:]
    train_gene_ids = [x.decode("utf-8") for x in f.root.train_cite_inputs.axis0[:]]
    train_cell_ids = [x.decode("utf-8") for x in f.root.train_cite_inputs.axis1[:]]
with tb.File("train_cite_targets.h5", "r") as f:
    y_train = f.root.train_cite_targets.block0_values[:]
    target_ids = [x.decode("utf-8") for x in f.root.train_cite_targets.axis0[:]]
    target_cell_ids = [x.decode("utf-8") for x in f.root.train_cite_targets.axis1[:]]

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

if train_cell_ids != target_cell_ids:
    raise ValueError("Training input cells and target cells are not aligned.")


# 2. Remove corrupted donor 27678 cells from TRAINING only
# These cells are duplicates of donor 32606 training data
# Kept in test set for submission integrity

print("Removing corrupted donor 27678 cells from training...")
metadata = pd.read_csv("metadata.csv")
bad_train_cells = set(metadata[
    (metadata["donor"] == 27678) &
    (metadata["day"] == 2)
]["cell_id"].tolist())
clean_idx = [i for i, c in enumerate(train_cell_ids) if c not in bad_train_cells]
X_train = X_train[clean_idx]
y_train = y_train[clean_idx]
train_cell_ids = [train_cell_ids[i] for i in clean_idx]
target_cell_ids = [target_cell_ids[i] for i in clean_idx]
print(f"Training cells after removing corrupted donor: {X_train.shape[0]}")


# 3. HVG selection on full training set
# HVG accounts for mean-variance relationship — selects genes
# that are more variable than expected given their expression
# level, enriching for lineage-specific markers over
# constitutively expressed housekeeping genes

print(f"Selecting top {USE_TOP_GENES} highly variable genes (HVG)...")
adata_train = ad.AnnData(
    X=X_train,
    obs=pd.DataFrame(index=range(X_train.shape[0])),
    var=pd.DataFrame(index=train_gene_ids)
)
sc.pp.highly_variable_genes(
    adata_train,
    n_top_genes=USE_TOP_GENES,
    flavor="cell_ranger"
)
hvg_mask = adata_train.var["highly_variable"].values
top_idx = np.where(hvg_mask)[0]
selected_gene_ids = [train_gene_ids[i] for i in top_idx]
X_train_sel = X_train[:, top_idx]
print(f"Selected {X_train_sel.shape[1]} HVGs")
print("Selected training shape:", X_train_sel.shape)


# 4. Fit TruncatedSVD on full training set

print(f"Fitting TruncatedSVD with {N_COMPONENTS} components...")
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
X_train_svd = svd.fit_transform(X_train_sel)


# 5. Fit Ridge on full training set

print(f"Fitting Ridge(alpha={ALPHA})...")
ridge = Ridge(alpha=ALPHA)
ridge.fit(X_train_svd, y_train)


# 6. Load original test inputs

print("Loading original test inputs...")
with tb.File("test_cite_inputs.h5", "r") as f:
    X_test = f.root.test_cite_inputs.block0_values[:]
    test_gene_ids = [x.decode("utf-8") for x in f.root.test_cite_inputs.axis0[:]]
    test_cell_ids = [x.decode("utf-8") for x in f.root.test_cite_inputs.axis1[:]]
X_test = X_test.astype(np.float32)
print("Original X_test shape:", X_test.shape)


# 7. Flag corrupted donor 27678 cells in test set
# Kept for submission integrity — evaluation_ids.csv references
# them. Per Kaggle: predictions do not affect evaluation score.

bad_test_cells = set(metadata[
    (metadata["donor"] == 27678) &
    (metadata["day"] == 2)
]["cell_id"].tolist())
corrupted_in_test = [c for c in test_cell_ids if c in bad_test_cells]
print(f"Corrupted donor 27678 cells in test (kept for submission): {len(corrupted_in_test)}")


# 8. Load real donor 27678 test inputs
# File uses non-standard two-block HDF5 format (mixed dtypes)
# pd.read_hdf handles block merging automatically

print("Loading real donor 27678 test inputs...")
X_new_df = pd.read_hdf("test_cite_inputs_day_2_donor_27678.h5",
                        key="test_cite_inputs_day_2_donor_27678")
new_cell_ids = X_new_df.index.tolist()
new_axis0 = X_new_df.columns.tolist()
X_new = X_new_df.values.astype(np.float32)
print("Donor 27678 shape:", X_new.shape)

# Align donor 27678 genes to original test gene order
new_gene_to_idx = {g: i for i, g in enumerate(new_axis0)}
new_selected_idx = [new_gene_to_idx[g] for g in test_gene_ids]
X_new_aligned = X_new[:, new_selected_idx]
print("Aligned donor 27678 shape:", X_new_aligned.shape)


# 9. Concatenate original test + real donor 27678 cells

print("Concatenating test sets...")
X_test_combined = np.concatenate([X_test, X_new_aligned], axis=0)
test_cell_ids_combined = test_cell_ids + new_cell_ids
print(f"Combined test shape: {X_test_combined.shape}")


# 10. Align selected HVG genes to test set

print("Aligning selected HVG genes between train and test...")
test_gene_to_idx = {g: i for i, g in enumerate(test_gene_ids)}
missing_genes = [g for g in selected_gene_ids if g not in test_gene_to_idx]
if missing_genes:
    raise ValueError(f"{len(missing_genes)} selected genes missing from test set.")
test_selected_idx = [test_gene_to_idx[g] for g in selected_gene_ids]
X_test_sel = X_test_combined[:, test_selected_idx]
print("Selected combined test shape:", X_test_sel.shape)


# 11. Project test data and predict

print("Projecting test data and predicting...")
X_test_svd = svd.transform(X_test_sel)
y_pred = ridge.predict(X_test_svd)
print("Prediction matrix shape:", y_pred.shape)


# 12. Convert predictions to long format

print("Converting predictions to long format...")
pred_df = pd.DataFrame(y_pred, index=test_cell_ids_combined, columns=target_ids)
pred_long = pred_df.stack().reset_index()
pred_long.columns = ["cell_id", "target_id", "target"]
print(pred_long.head())


# 13. Load evaluation_ids and filter to CITE targets

print("Loading evaluation_ids.csv...")
eval_ids = pd.read_csv("evaluation_ids.csv")
eval_ids_for_merge = eval_ids.rename(columns={"gene_id": "target_id"})
cite_target_set = set(target_ids)
eval_ids_cite = eval_ids_for_merge[
    eval_ids_for_merge["target_id"].isin(cite_target_set)
].copy()
print("Total evaluation rows:", len(eval_ids_for_merge))
print("CITE evaluation rows:", len(eval_ids_cite))


# 14. Merge predictions with evaluation IDs

print("Merging predictions with CITE evaluation IDs...")
submission_cite = eval_ids_cite.merge(
    pred_long,
    on=["cell_id", "target_id"],
    how="left"
)
missing_targets = submission_cite["target"].isna().sum()
print("Missing CITE targets after merge:", missing_targets)
if missing_targets != 0:
    print("Example missing rows:")
    print(submission_cite[submission_cite["target"].isna()].head())
    raise ValueError("CITE submission still contains missing target values.")

submission_cite = submission_cite[["row_id", "target"]].sort_values("row_id")
print(submission_cite.head())
print("CITE submission shape:", submission_cite.shape)
print(submission_cite["target"].describe())

submission_cite.to_csv("submission_cite_only.csv", index=False)
print("Saved submission_cite_only.csv")