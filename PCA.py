import numpy as np
import pandas as pd
import tables as tb
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
import time

start_time = time.time()


# Settings

SAMPLE_SIZE = 70988
USE_TOP_GENES = 2000
N_COMPONENTS = 100
ALPHA = 1.0
N_SPLITS = 5
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


def run_permutation_test(X_train_pca, y_train, X_val_pca, y_val, alpha, random_seed):
    y_train_perm = y_train.copy()
    rng = np.random.default_rng(random_seed)

    for j in range(y_train_perm.shape[1]):
        rng.shuffle(y_train_perm[:, j])

    ridge_perm = Ridge(alpha=alpha)
    ridge_perm.fit(X_train_pca, y_train_perm)
    y_pred_perm = ridge_perm.predict(X_val_pca)

    return compute_score(y_val, y_pred_perm)


# 1. Load cell IDs from HDF5

print("Loading cell IDs...")
with tb.File("train_cite_inputs.h5", "r") as f:
    cell_ids_bytes = f.root.train_cite_inputs.axis1[:]
cell_ids = [cid.decode("utf-8") for cid in cell_ids_bytes]
n_total = len(cell_ids)
print(f"Total cells: {n_total}")


# 2. Load metadata and align to HDF5 order

print("Loading metadata...")
metadata = pd.read_csv("metadata.csv")
metadata_filtered = metadata[metadata["cell_id"].isin(cell_ids)].copy()
print(f"Metadata rows after filtering: {len(metadata_filtered)}")
assert len(metadata_filtered) == n_total, "Metadata row count does not match HDF5 cell count."

metadata_filtered = metadata_filtered.set_index("cell_id").loc[cell_ids].reset_index()

days = metadata_filtered["day"].to_numpy()
donors = metadata_filtered["donor"].to_numpy()

print("Unique days:", np.unique(days))
print("Unique donors:", np.unique(donors))
print("\nCell counts by day and donor:")
print(metadata_filtered.groupby(["day", "donor"]).size())


# 3. Restrict to CITE training days and sample cells

print("\nSampling cells...")
np.random.seed(RANDOM_SEED)

valid_days = [2, 3, 4]
valid_mask = metadata_filtered["day"].isin(valid_days).to_numpy()
valid_idx = np.where(valid_mask)[0]

days_valid = days[valid_idx]
donors_valid = donors[valid_idx]

sample_size = min(SAMPLE_SIZE, len(valid_idx))

day_indices = {day: valid_idx[days_valid == day] for day in valid_days}

sample_idx = []
for day in valid_days:
    idx_day = day_indices[day]
    n_avail = len(idx_day)
    if n_avail == 0:
        continue

    n_sample_day = int(sample_size * n_avail / len(valid_idx))
    if n_sample_day == 0:
        n_sample_day = 1
    n_sample_day = min(n_sample_day, n_avail)

    chosen = np.random.choice(idx_day, n_sample_day, replace=False)
    sample_idx.extend(chosen)

sample_idx = np.array(sorted(sample_idx))
days_sampled = days[sample_idx]
donors_sampled = donors[sample_idx]

print(
    f"Sampled {len(sample_idx)} cells "
    f"(day 2: {np.sum(days_sampled == 2)}, "
    f"day 3: {np.sum(days_sampled == 3)}, "
    f"day 4: {np.sum(days_sampled == 4)})"
)

print("\nSampled cell counts by day and donor:")
sampled_meta = pd.DataFrame({
    "day": days_sampled,
    "donor": donors_sampled
})
print(sampled_meta.groupby(["day", "donor"]).size())


# 4. Load RNA and protein data for sampled cells

print("\nLoading RNA subset...")
with tb.File("train_cite_inputs.h5", "r") as f:
    X = f.root.train_cite_inputs.block0_values[sample_idx, :]

print("Loading protein subset...")
with tb.File("train_cite_targets.h5", "r") as f:
    y = f.root.train_cite_targets.block0_values[sample_idx, :]

X = X.astype(np.float32)
y = y.astype(np.float32)

print(f"\nLoaded X shape: {X.shape}")
print(f"Loaded y shape: {y.shape}")


# 5. Build grouped CV labels

groups = np.array([f"{donor}_{day}" for donor, day in zip(donors_sampled, days_sampled)])

unique_groups = np.unique(groups)
print(f"\nNumber of unique donor-day groups: {len(unique_groups)}")
print("Unique groups:", unique_groups)

if len(unique_groups) < N_SPLITS:
    raise ValueError(
        f"Only {len(unique_groups)} unique groups are available, "
        f"but N_SPLITS={N_SPLITS}. Reduce N_SPLITS."
    )

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

    
    # Top variable genes selected on TRAIN only
    
    if USE_TOP_GENES is not None:
        var = X_train.var(axis=0)
        top_idx = np.argsort(var)[-USE_TOP_GENES:]
        X_train_fold = X_train[:, top_idx]
        X_val_fold = X_val[:, top_idx]
        print(f"Using top {USE_TOP_GENES} variable genes")
    else:
        X_train_fold = X_train
        X_val_fold = X_val
        print("Using all genes")

    
    # PCA fit on TRAIN only
    
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_fold)
    X_val_pca = pca.transform(X_val_fold)
    print("PCA done.")

    
    # Ridge fit on TRAIN only
    
    ridge = Ridge(alpha=ALPHA)
    ridge.fit(X_train_pca, y_train)
    y_pred = ridge.predict(X_val_pca)

    fold_score = compute_score(y_val, y_pred)
    fold_scores.append(fold_score)
    print(f"Fold {fold} model score: {fold_score:.6f}")

    
    # Dummy baseline
    
    dummy_score = run_dummy_baseline(y_train, y_val)
    fold_dummy_scores.append(dummy_score)
    print(f"Fold {fold} dummy score: {dummy_score:.6f}")

    
    # Permutation test
    
    perm_score = run_permutation_test(
        X_train_pca=X_train_pca,
        y_train=y_train,
        X_val_pca=X_val_pca,
        y_val=y_val,
        alpha=ALPHA,
        random_seed=RANDOM_SEED + fold
    )
    fold_perm_scores.append(perm_score)
    print(f"Fold {fold} permutation score: {perm_score:.6f}")


# 7. Final summary

print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

for i in range(N_SPLITS):
    print(
        f"Fold {i+1}: "
        f"model={fold_scores[i]:.6f}, "
        f"dummy={fold_dummy_scores[i]:.6f}, "
        f"perm={fold_perm_scores[i]:.6f}"
    )

print("\nAverage model score:       {:.6f}".format(np.mean(fold_scores)))
print("Average dummy score:       {:.6f}".format(np.mean(fold_dummy_scores)))
print("Average permutation score: {:.6f}".format(np.mean(fold_perm_scores)))

print("\nStd model score:           {:.6f}".format(np.std(fold_scores)))
print("Std dummy score:           {:.6f}".format(np.std(fold_dummy_scores)))
print("Std permutation score:     {:.6f}".format(np.std(fold_perm_scores)))

print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")