#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:49:49 2026

@author: bass
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom

# ============================================================
# 0. PATHS
# ============================================================

BASE_DIR = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/fearConditioningTrial"

behavior_file = os.path.join(BASE_DIR, "fear_conditioning_behavior_phenotypes.csv")
overlap_file = os.path.join(BASE_DIR, "true_overlap_between_connectome_and_fear_conditioning_5-12.csv")

CONNECTOME_DIR = "/Users/bass/LaurenOh/data/connectomes"

# Optional: put AD risk network matrices here.
# These should be 324 x 324 CSV matrices, where nonzero entries = risk-factor edges.
RISK_NETWORK_DIR = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/risk_network_matrices"

OUT_DIR = os.path.join(BASE_DIR, "NBS_overlap_results")
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

N_PERM = 1000
P_EDGE_THRESHOLD = 0.001
RANDOM_SEED = 123

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
# 3. LOAD CONNECTOMES
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

connectomes = []
rows = []

for _, r in df.iterrows():
    fpath = os.path.join(CONNECTOME_DIR, r["Connectome_File"])

    if os.path.exists(fpath):
        try:
            connectomes.append(load_connectome(fpath))
            rows.append(r)
        except Exception as e:
            print("Skipping:", fpath)
            print(e)

df = pd.DataFrame(rows).reset_index(drop=True)
connectomes = np.array(connectomes)

print("Loaded connectomes:", connectomes.shape)

if connectomes.shape[0] == 0:
    raise RuntimeError("No connectomes loaded.")

# ============================================================
# 4. VECTORIZE AND CLEAN EDGES
# ============================================================

n_nodes = connectomes.shape[1]
triu_idx = np.triu_indices(n_nodes, k=1)

edge_i_all = triu_idx[0]
edge_j_all = triu_idx[1]

X_edges = np.array([m[triu_idx] for m in connectomes])

print("Initial edges:", X_edges.shape[1])
print("NaNs before cleaning:", np.isnan(X_edges).sum())

X_edges[~np.isfinite(X_edges)] = np.nan

valid_edges = np.isfinite(X_edges).all(axis=0)
edge_sd = np.nanstd(X_edges, axis=0)
valid_edges = valid_edges & (edge_sd > 0)

X_edges = X_edges[:, valid_edges]
edge_i = edge_i_all[valid_edges]
edge_j = edge_j_all[valid_edges]

print("Cleaned edges:", X_edges.shape[1])

# ============================================================
# 5. DESIGN / RESIDUALIZATION
# ============================================================

def make_covariate_matrix(data, covariates):
    C = data[covariates].copy()

    for col in C.columns:
        if C[col].dtype == "object":
            C[col] = C[col].astype(str)

    C = pd.get_dummies(C, drop_first=True)
    C = C.apply(pd.to_numeric, errors="coerce")

    X = np.column_stack([np.ones(C.shape[0]), C.values.astype(float)])
    return X

def residualize(Y, Cov):
    beta = np.linalg.pinv(Cov) @ Y
    return Y - Cov @ beta

def compute_t_values(y_res, X_res):
    y = y_res.copy()
    X = X_res.copy()

    y = y - y.mean()
    X = X - X.mean(axis=0)

    y_sd = y.std(ddof=1)
    X_sd = X.std(axis=0, ddof=1)

    good = (X_sd > 0) & np.isfinite(X_sd)

    r = np.full(X.shape[1], np.nan)
    Xz = X[:, good] / X_sd[good]
    yz = y / y_sd

    r_good = np.dot(yz, Xz) / (len(y) - 1)
    r[good] = r_good

    r = np.clip(r, -0.999999, 0.999999)
    dof = len(y) - 2
    tvals = r * np.sqrt(dof / (1 - r ** 2))

    return tvals

# ============================================================
# 6. NBS COMPONENT FUNCTIONS
# ============================================================

def find_components(edge_i, edge_j, suprathreshold_mask, n_nodes):
    selected_edges = np.where(suprathreshold_mask)[0]

    adjacency = {i: [] for i in range(n_nodes)}

    for edge_idx in selected_edges:
        a = int(edge_i[edge_idx])
        b = int(edge_j[edge_idx])
        adjacency[a].append((b, edge_idx))
        adjacency[b].append((a, edge_idx))

    visited_nodes = set()
    components = []

    for node in range(n_nodes):
        if node in visited_nodes:
            continue

        stack = [node]
        comp_nodes = set()
        comp_edges = set()

        while stack:
            current = stack.pop()

            if current in visited_nodes:
                continue

            visited_nodes.add(current)
            comp_nodes.add(current)

            for neighbor, edge_idx in adjacency[current]:
                comp_edges.add(edge_idx)
                if neighbor not in visited_nodes:
                    stack.append(neighbor)

        if len(comp_edges) > 0:
            components.append({
                "nodes": sorted(list(comp_nodes)),
                "edges": sorted(list(comp_edges)),
                "size_edges": len(comp_edges),
                "size_nodes": len(comp_nodes)
            })

    components = sorted(components, key=lambda x: x["size_edges"], reverse=True)

    return components

def edge_mask_to_matrix(mask, values=None):
    mat = np.zeros((n_nodes, n_nodes))

    if values is None:
        vals = mask.astype(float)
    else:
        vals = values * mask.astype(float)

    mat[edge_i, edge_j] = vals
    mat = mat + mat.T

    return mat

# ============================================================
# 7. RISK NETWORK LOADING AND OVERLAP
# ============================================================

def load_risk_networks():
    risk_files = glob.glob(os.path.join(RISK_NETWORK_DIR, "*.csv"))

    risk_networks = {}

    if len(risk_files) == 0:
        print("\nNo risk network matrices found. Skipping overlap step.")
        return risk_networks

    print("\nRisk networks found:", len(risk_files))

    for f in risk_files:
        name = os.path.splitext(os.path.basename(f))[0]

        try:
            mat = pd.read_csv(f, header=None).values.astype(float)
        except Exception:
            mat = pd.read_csv(f, index_col=0).values.astype(float)

        if mat.shape != (n_nodes, n_nodes):
            print("Skipping risk matrix with wrong shape:", name, mat.shape)
            continue

        mat = np.nan_to_num(mat)
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 0)

        risk_vec_full = mat[triu_idx]
        risk_vec_clean = risk_vec_full[valid_edges]

        risk_mask = risk_vec_clean != 0

        risk_networks[name] = risk_mask

        print(name, "edges:", int(risk_mask.sum()))

    return risk_networks

def compute_overlap(component_mask, risk_mask):
    total_edges = len(component_mask)
    comp_edges = int(component_mask.sum())
    risk_edges = int(risk_mask.sum())
    overlap_edges = int((component_mask & risk_mask).sum())

    if comp_edges == 0 or risk_edges == 0:
        pval = np.nan
        enrichment = np.nan
    else:
        expected = comp_edges * risk_edges / total_edges
        enrichment = overlap_edges / expected if expected > 0 else np.nan
        pval = hypergeom.sf(overlap_edges - 1, total_edges, risk_edges, comp_edges)

    return overlap_edges, comp_edges, risk_edges, enrichment, pval

risk_networks = load_risk_networks()

# ============================================================
# 8. RUN NBS
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

all_overlap_rows = []

for phenotype in phenotypes:

    print("\n================================================")
    print("NBS phenotype:", phenotype)
    print("================================================")

    cols = ["Animal ID", phenotype] + covariates
    d = df[cols].copy()

    valid_subjects = d.notna().all(axis=1).values
    d = d.loc[valid_subjects].reset_index(drop=True)
    X_use = X_edges[valid_subjects, :]

    y = d[phenotype].astype(float).values
    Cov = make_covariate_matrix(d, covariates)

    valid_numeric = np.isfinite(Cov).all(axis=1) & np.isfinite(y)
    d = d.loc[valid_numeric].reset_index(drop=True)
    X_use = X_use[valid_numeric, :]
    y = y[valid_numeric]
    Cov = Cov[valid_numeric, :]

    print("N used:", len(y))

    y_res = residualize(y, Cov)
    X_res = residualize(X_use, Cov)

    X_res = X_res.astype(float)
    X_res[~np.isfinite(X_res)] = 0

    t_obs = compute_t_values(y_res, X_res)

    dof = len(y_res) - 2
    t_thresh = stats.t.ppf(1 - P_EDGE_THRESHOLD / 2, dof)

    supra_obs = np.abs(t_obs) > t_thresh

    components = find_components(edge_i, edge_j, supra_obs, n_nodes)

    print("Edge threshold p:", P_EDGE_THRESHOLD)
    print("T threshold:", t_thresh)
    print("Suprathreshold edges:", int(supra_obs.sum()))
    print("Observed components:", len(components))

    # Permutation max component size
    max_sizes = np.zeros(N_PERM)

    for p in range(N_PERM):
        y_perm = rng.permutation(y_res)
        t_perm = compute_t_values(y_perm, X_res)
        supra_perm = np.abs(t_perm) > t_thresh
        comps_perm = find_components(edge_i, edge_j, supra_perm, n_nodes)

        if len(comps_perm) > 0:
            max_sizes[p] = comps_perm[0]["size_edges"]
        else:
            max_sizes[p] = 0

        if (p + 1) % 100 == 0:
            print("Permutation", p + 1, "/", N_PERM)

    pheno_dir = os.path.join(OUT_DIR, phenotype)
    os.makedirs(pheno_dir, exist_ok=True)

    np.savetxt(
        os.path.join(pheno_dir, "permutation_max_component_sizes.csv"),
        max_sizes,
        delimiter=","
    )

    # Component summaries
    summary_rows = []

    for ci, comp in enumerate(components, start=1):
        comp_mask = np.zeros(X_edges.shape[1], dtype=bool)
        comp_mask[comp["edges"]] = True

        p_nbs = np.mean(max_sizes >= comp["size_edges"])

        summary_rows.append({
            "phenotype": phenotype,
            "component": ci,
            "size_edges": comp["size_edges"],
            "size_nodes": comp["size_nodes"],
            "p_nbs": p_nbs
        })

        # Save component edge list
        comp_edges_df = pd.DataFrame({
            "node_i": edge_i[comp["edges"]],
            "node_j": edge_j[comp["edges"]],
            "t_value": t_obs[comp["edges"]]
        })

        comp_edges_df.to_csv(
            os.path.join(pheno_dir, f"component_{ci}_edges.csv"),
            index=False
        )

        # Save binary matrix
        comp_mat = edge_mask_to_matrix(comp_mask)
        pd.DataFrame(comp_mat).to_csv(
            os.path.join(pheno_dir, f"component_{ci}_binary_matrix.csv"),
            index=False,
            header=False
        )

        # Save t matrix for component
        comp_t_mat = edge_mask_to_matrix(comp_mask, values=t_obs)
        pd.DataFrame(comp_t_mat).to_csv(
            os.path.join(pheno_dir, f"component_{ci}_t_matrix.csv"),
            index=False,
            header=False
        )

        # Overlap with AD risk networks
        for risk_name, risk_mask in risk_networks.items():
            overlap_edges, comp_edges, risk_edges, enrichment, p_overlap = compute_overlap(
                comp_mask,
                risk_mask
            )

            all_overlap_rows.append({
                "phenotype": phenotype,
                "component": ci,
                "risk_network": risk_name,
                "component_edges": comp_edges,
                "risk_edges": risk_edges,
                "overlap_edges": overlap_edges,
                "enrichment": enrichment,
                "p_overlap_hypergeom": p_overlap,
                "p_nbs": p_nbs
            })

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.shape[0] > 0:
        summary_df.to_csv(
            os.path.join(pheno_dir, "NBS_component_summary.csv"),
            index=False
        )

        print("\nTop components:")
        print(summary_df.head(10))
    else:
        print("No NBS components found.")

# ============================================================
# 9. SAVE OVERLAP SUMMARY
# ============================================================

if len(all_overlap_rows) > 0:
    overlap_df = pd.DataFrame(all_overlap_rows)
    overlap_df.to_csv(
        os.path.join(OUT_DIR, "NBS_AD_risk_overlap_summary.csv"),
        index=False
    )

    print("\nSaved overlap summary:")
    print(os.path.join(OUT_DIR, "NBS_AD_risk_overlap_summary.csv"))
else:
    print("\nNo overlap results saved because no risk matrices were found.")

print("\nDONE.")
print("Results saved to:")
print(OUT_DIR)