#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# 0. PATHS
# ============================================================

BASE_DIR = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/fearConditioningTrial"

behavior_file = os.path.join(BASE_DIR, "fear_conditioning_behavior_phenotypes.csv")
overlap_file = os.path.join(BASE_DIR, "true_overlap_between_connectome_and_fear_conditioning_5-12.csv")

CONNECTOME_DIR = "/Users/bass/LaurenOh/data/connectomes"

OUT_DIR = os.path.join(BASE_DIR, "edgewise_GLM_results_clean")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. SETTINGS
# ============================================================

phenotypes = [
    "PHENO_learning",
    "PHENO_context_memory",
    "PHENO_cue_memory_vs_exploration"
]

covariates = [
    "Sex",
    "Genotype",
    "Diet",
    "Lifestyle",
    "Age_Months"
]

# ============================================================
# 2. LOAD DATA
# ============================================================

behavior = pd.read_csv(behavior_file)
overlap = pd.read_csv(overlap_file)

behavior["Animal ID"] = behavior["Animal ID"].astype(str).str.strip()
overlap["Animal ID"] = overlap["Animal ID"].astype(str).str.strip()

overlap = overlap.drop_duplicates(subset="Animal ID")

df = overlap[["Animal ID", "Connectome_File"]].merge(
    behavior,
    on="Animal ID",
    how="inner"
)

print("Merged dataset size:", df.shape[0])

# ============================================================
# 3. CONNECTOME LOADER
# ============================================================

def load_connectome(path):
    try:
        mat = pd.read_csv(path, header=None).values.astype(float)
    except Exception:
        mat = pd.read_csv(path, index_col=0).values.astype(float)

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Not square: {path}, shape={mat.shape}")

    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat, 0)

    return mat

# ============================================================
# 4. LOAD ONLY RELEVANT CONNECTOMES
# ============================================================

connectomes = []
rows = []
missing = []

for _, r in df.iterrows():
    fpath = os.path.join(CONNECTOME_DIR, r["Connectome_File"])

    if not os.path.exists(fpath):
        missing.append(fpath)
        continue

    try:
        mat = load_connectome(fpath)
        connectomes.append(mat)
        rows.append(r)
    except Exception as e:
        print("Skipping:", fpath)
        print(e)

df = pd.DataFrame(rows).reset_index(drop=True)
connectomes = np.array(connectomes)

print("Loaded connectomes:", connectomes.shape)

if len(missing) > 0:
    print("Missing connectome files:", len(missing))
    print(missing[:10])

if connectomes.shape[0] == 0:
    raise RuntimeError("No connectomes loaded. Check CONNECTOME_DIR and Connectome_File names.")

# ============================================================
# 5. VECTORIZE EDGES
# ============================================================

n_nodes = connectomes.shape[1]
triu_idx = np.triu_indices(n_nodes, k=1)

edge_i = triu_idx[0]
edge_j = triu_idx[1]

X_edges = np.array([m[triu_idx] for m in connectomes])

print("Nodes:", n_nodes)
print("Initial edges:", X_edges.shape[1])

# ============================================================
# 6. CLEAN EDGE MATRIX BEFORE GLM
# ============================================================

print("\nBefore cleaning X_edges:")
print("NaNs:", np.isnan(X_edges).sum())
print("Infs:", np.isinf(X_edges).sum())

X_edges[~np.isfinite(X_edges)] = np.nan

valid_edges = np.isfinite(X_edges).all(axis=0)

edge_sd = np.nanstd(X_edges, axis=0)
valid_edges = valid_edges & (edge_sd > 0)

print("Valid edges retained:", int(valid_edges.sum()), "out of", X_edges.shape[1])

X_edges = X_edges[:, valid_edges]
edge_i = edge_i[valid_edges]
edge_j = edge_j[valid_edges]

print("Cleaned edge matrix shape:", X_edges.shape)

# ============================================================
# 7. MAKE DESIGN MATRIX
# ============================================================

def make_covariate_matrix(data, covariates):
    C = data[covariates].copy()

    for col in C.columns:
        if C[col].dtype == "object":
            C[col] = C[col].astype(str)

    C = pd.get_dummies(C, drop_first=True)
    C = C.apply(pd.to_numeric, errors="coerce")

    X = np.column_stack([np.ones(C.shape[0]), C.values.astype(float)])
    names = ["Intercept"] + list(C.columns)

    return X, names

# ============================================================
# 8. GLM FUNCTION
# ============================================================

def fit_edge_glm(y, edge, Cov):
    """
    Model:
        y ~ intercept + covariates + edge
    """

    if not np.isfinite(y).all():
        return np.nan, np.nan, np.nan

    if not np.isfinite(edge).all():
        return np.nan, np.nan, np.nan

    if np.nanstd(edge) == 0:
        return np.nan, np.nan, np.nan

    X = np.column_stack([Cov, edge])

    if not np.isfinite(X).all():
        return np.nan, np.nan, np.nan

    n, p = X.shape
    dof = n - p

    if dof <= 0:
        return np.nan, np.nan, np.nan

    try:
        beta = np.linalg.pinv(X) @ y

        y_hat = X @ beta
        resid = y - y_hat

        rss = np.sum(resid ** 2)
        sigma2 = rss / dof

        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * XtX_inv))

        beta_edge = beta[-1]
        se_edge = se[-1]

        if se_edge == 0 or not np.isfinite(se_edge):
            return beta_edge, np.nan, np.nan

        t_edge = beta_edge / se_edge
        p_edge = 2 * stats.t.sf(np.abs(t_edge), dof)

        return beta_edge, t_edge, p_edge

    except Exception:
        return np.nan, np.nan, np.nan

# ============================================================
# 9. FDR FUNCTION
# ============================================================

def fdr_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)

    qvals = np.full_like(pvals, np.nan, dtype=float)
    reject = np.zeros_like(pvals, dtype=bool)

    valid = np.isfinite(pvals)
    p = pvals[valid]

    if len(p) == 0:
        return reject, qvals

    order = np.argsort(p)
    ranked_p = p[order]
    m = len(p)

    q = ranked_p * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q[q > 1] = 1

    q_unsorted = np.empty_like(q)
    q_unsorted[order] = q

    qvals[valid] = q_unsorted
    reject[valid] = q_unsorted < alpha

    return reject, qvals

# ============================================================
# 10. RUN EDGE-WISE GLM
# ============================================================

def run_glm(df, X_edges, phenotype):

    cols = ["Animal ID", phenotype] + covariates
    d = df[cols].copy()

    valid_subjects = d.notna().all(axis=1).values
    d = d.loc[valid_subjects].reset_index(drop=True)
    X_use = X_edges[valid_subjects, :]

    print("\n======================================")
    print("Running:", phenotype)
    print("N:", len(d))
    print("Edges:", X_use.shape[1])
    print("======================================")

    y = d[phenotype].astype(float).values

    Cov, cov_names = make_covariate_matrix(d, covariates)

    valid_cov = np.isfinite(Cov).all(axis=1) & np.isfinite(y)

    d = d.loc[valid_cov].reset_index(drop=True)
    y = y[valid_cov]
    Cov = Cov[valid_cov, :]
    X_use = X_use[valid_cov, :]

    print("N after numeric covariate cleaning:", len(d))

    betas = np.full(X_use.shape[1], np.nan)
    tvals = np.full(X_use.shape[1], np.nan)
    pvals = np.full(X_use.shape[1], np.nan)

    for e in range(X_use.shape[1]):
        beta, tval, pval = fit_edge_glm(y, X_use[:, e], Cov)
        betas[e] = beta
        tvals[e] = tval
        pvals[e] = pval

    reject, qvals = fdr_bh(pvals, alpha=0.05)

    print("Nominal p < 0.05:", int(np.sum(pvals < 0.05)))
    print("FDR q < 0.05:", int(np.sum(reject)))

    results = pd.DataFrame({
        "node_i": edge_i,
        "node_j": edge_j,
        "beta": betas,
        "t": tvals,
        "p": pvals,
        "q_fdr": qvals,
        "significant_fdr_0.05": reject
    })

    return results, d

# ============================================================
# 11. RUN ALL PHENOTYPES AND SAVE
# ============================================================

for pheno in phenotypes:

    results, used = run_glm(df, X_edges, pheno)

    out_edges = os.path.join(OUT_DIR, f"{pheno}_edges.csv")
    results.to_csv(out_edges, index=False)

    beta_mat = np.zeros((n_nodes, n_nodes))
    beta_mat[edge_i, edge_j] = results["beta"].values
    beta_mat = beta_mat + beta_mat.T

    out_beta = os.path.join(OUT_DIR, f"{pheno}_beta_matrix.csv")
    pd.DataFrame(beta_mat).to_csv(out_beta, index=False, header=False)

    sig_beta_mat = np.zeros((n_nodes, n_nodes))
    sig_beta_mat[edge_i, edge_j] = (
        results["beta"].values * results["significant_fdr_0.05"].values
    )
    sig_beta_mat = sig_beta_mat + sig_beta_mat.T

    out_sig = os.path.join(OUT_DIR, f"{pheno}_FDR_significant_beta_matrix.csv")
    pd.DataFrame(sig_beta_mat).to_csv(out_sig, index=False, header=False)

    out_subjects = os.path.join(OUT_DIR, f"{pheno}_subjects_used.csv")
    used.to_csv(out_subjects, index=False)

    print("Saved:")
    print(out_edges)
    print(out_beta)
    print(out_sig)
    print(out_subjects)

# ============================================================
# 12. SAVE COHORT
# ============================================================

cohort_file = os.path.join(OUT_DIR, "analysis_cohort_loaded_connectomes.csv")
df.to_csv(cohort_file, index=False)

print("\nDONE.")
print("Results saved to:")
print(OUT_DIR)