import numpy as np
import tables as tb


print("Loading data...")

with tb.File("train_cite_inputs.h5", "r") as f:
    X = f.root.train_cite_inputs.block0_values[:]

X = X.astype(np.float32)

print("Shape:", X.shape)


# 1. Overall sparsity

zero_fraction = (X == 0).mean()
print(f"\nOverall zero fraction: {zero_fraction:.4f}")


# 2. Per-gene sparsity

gene_zero_frac = (X == 0).mean(axis=0)

print("\nPer-gene zero fraction stats:")
print(f"Min:  {gene_zero_frac.min():.4f}")
print(f"Mean: {gene_zero_frac.mean():.4f}")
print(f"Max:  {gene_zero_frac.max():.4f}")


# 3. Value distribution

print("\nValue stats:")
print(f"Min:  {X.min():.4f}")
print(f"Mean: {X.mean():.4f}")
print(f"Max:  {X.max():.4f}")
print(f"Std:  {X.std():.4f}")


# 4. Histogram (non-zero values)

nonzero = X[X > 0]




# 5. Gene variance check

gene_var = X.var(axis=0)

print("\nGene variance stats:")
print(f"Min:  {gene_var.min():.6f}")
print(f"Mean: {gene_var.mean():.6f}")
print(f"Max:  {gene_var.max():.6f}")

# Top variable genes
top_var = np.sort(gene_var)[-10:]
print("\nTop 10 gene variances:")
print(top_var)


# 6. Extremely sparse genes

very_sparse = np.sum(gene_zero_frac > 0.99)
print(f"\nGenes with >99% zeros: {very_sparse} / {X.shape[1]}")


# 7. Extremely dense genes

very_dense = np.sum(gene_zero_frac < 0.1)
print(f"Genes with <10% zeros: {very_dense} / {X.shape[1]}")