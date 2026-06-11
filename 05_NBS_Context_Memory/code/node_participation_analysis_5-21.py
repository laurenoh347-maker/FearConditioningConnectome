#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# ============================================================
# 0. PATHS
# ============================================================

BASE_DIR = "/Users/bass/LaurenOh/Connectome:Fear Conditioning"

# RENAMED FOLDER:
# NBS_results_pthreshold0.001_5-21

NBS_DIR = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/NBS_results_pthreshold0.001_5-21"

OUT_DIR = os.path.join(
    BASE_DIR,
    "NBS_node_participation_results"
)

os.makedirs(OUT_DIR, exist_ok=True)

PHENOTYPE = "PHENO_context_memory"
P_THRESHOLD = "0.001"

# ============================================================
# 1. FILES
# ============================================================

component_file = os.path.join(
    NBS_DIR,
    f"NBS_components_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

edge_table_file = os.path.join(
    NBS_DIR,
    f"NBS_edge_table_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

print("Component file:")
print(component_file)

print("\nEdge table file:")
print(edge_table_file)

# ============================================================
# 2. LOAD NBS OUTPUTS
# ============================================================

components = pd.read_csv(component_file)
edges = pd.read_csv(edge_table_file)

print("\nLoaded components:", components.shape)
print("Loaded edge table:", edges.shape)

if components.empty:
    raise RuntimeError("No NBS components found.")

# ============================================================
# 3. KEEP SUPRATHRESHOLD EDGES ONLY
# ============================================================

if "suprathreshold" in edges.columns:
    supra_edges = edges[edges["suprathreshold"] == True].copy()
else:
    supra_edges = edges[edges["p"] < float(P_THRESHOLD)].copy()

print("\nSuprathreshold edges:", supra_edges.shape[0])

# ============================================================
# 4. NODE PARTICIPATION ACROSS ALL SUPRATHRESHOLD EDGES
# ============================================================

all_nodes = np.concatenate([
    supra_edges["node_i"].values,
    supra_edges["node_j"].values
])

node_counts = pd.Series(all_nodes).value_counts().reset_index()

node_counts.columns = [
    "node_id",
    "n_suprathreshold_edges"
]

node_counts["node_id"] = node_counts["node_id"].astype(int)

node_counts = node_counts.sort_values(
    by="n_suprathreshold_edges",
    ascending=False
).reset_index(drop=True)

node_counts["rank"] = np.arange(
    1,
    len(node_counts) + 1
)

print("\nTop participating nodes across all suprathreshold edges:")
print(node_counts.head(20))

# ============================================================
# 5. NODE PARTICIPATION BY COMPONENT
# ============================================================

component_node_tables = []

for idx, row in components.iterrows():

    comp_rank = idx + 1
    component_id = row["component_id"]

    nodes = [
        int(x)
        for x in str(row["nodes"]).split(",")
        if str(x).strip() != ""
    ]

    comp_edges = supra_edges[
        supra_edges["node_i"].isin(nodes) &
        supra_edges["node_j"].isin(nodes)
    ].copy()

    if comp_edges.empty:
        continue

    comp_nodes = np.concatenate([
        comp_edges["node_i"].values,
        comp_edges["node_j"].values
    ])

    comp_counts = pd.Series(comp_nodes).value_counts().reset_index()

    comp_counts.columns = [
        "node_id",
        "n_edges_in_component"
    ]

    comp_counts["node_id"] = comp_counts["node_id"].astype(int)

    comp_counts["component_rank"] = comp_rank
    comp_counts["component_id"] = component_id
    comp_counts["component_n_nodes"] = row["n_nodes"]
    comp_counts["component_n_edges"] = row["n_edges"]

    if "p_nbs_size" in row.index:
        comp_counts["p_nbs_size"] = row["p_nbs_size"]

    if "p_nbs_mass" in row.index:
        comp_counts["p_nbs_mass"] = row["p_nbs_mass"]

    comp_counts = comp_counts.sort_values(
        by="n_edges_in_component",
        ascending=False
    )

    component_node_tables.append(comp_counts)

if len(component_node_tables) > 0:
    component_nodes = pd.concat(
        component_node_tables,
        ignore_index=True
    )
else:
    component_nodes = pd.DataFrame()

print("\nTop nodes in top component:")

if not component_nodes.empty:
    print(
        component_nodes[
            component_nodes["component_rank"] == 1
        ].head(20)
    )

# ============================================================
# 6. POSITIVE VS NEGATIVE EDGE PARTICIPATION
# ============================================================

positive_edges = supra_edges[
    supra_edges["t"] > 0
].copy()

negative_edges = supra_edges[
    supra_edges["t"] < 0
].copy()

def count_nodes(edge_df, count_name):

    if edge_df.empty:
        return pd.DataFrame(
            columns=["node_id", count_name]
        )

    nodes = np.concatenate([
        edge_df["node_i"].values,
        edge_df["node_j"].values
    ])

    out = pd.Series(nodes).value_counts().reset_index()

    out.columns = [
        "node_id",
        count_name
    ]

    out["node_id"] = out["node_id"].astype(int)

    return out

pos_counts = count_nodes(
    positive_edges,
    "n_positive_edges"
)

neg_counts = count_nodes(
    negative_edges,
    "n_negative_edges"
)

node_direction = node_counts.merge(
    pos_counts,
    on="node_id",
    how="left"
)

node_direction = node_direction.merge(
    neg_counts,
    on="node_id",
    how="left"
)

node_direction["n_positive_edges"] = (
    node_direction["n_positive_edges"]
    .fillna(0)
    .astype(int)
)

node_direction["n_negative_edges"] = (
    node_direction["n_negative_edges"]
    .fillna(0)
    .astype(int)
)

node_direction["positive_minus_negative"] = (
    node_direction["n_positive_edges"]
    -
    node_direction["n_negative_edges"]
)

# ============================================================
# 7. SAVE OUTPUTS
# ============================================================

out_all_nodes = os.path.join(
    OUT_DIR,
    f"node_participation_all_suprathreshold_edges_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

out_node_direction = os.path.join(
    OUT_DIR,
    f"node_participation_positive_negative_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

out_component_nodes = os.path.join(
    OUT_DIR,
    f"node_participation_by_component_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

node_counts.to_csv(
    out_all_nodes,
    index=False
)

node_direction.to_csv(
    out_node_direction,
    index=False
)

if not component_nodes.empty:
    component_nodes.to_csv(
        out_component_nodes,
        index=False
    )

# ============================================================
# 8. SAVE TOP COMPONENT EDGE LIST
# ============================================================

top_component = components.iloc[0]

top_nodes = [
    int(x)
    for x in str(top_component["nodes"]).split(",")
    if str(x).strip() != ""
]

top_edges = supra_edges[
    supra_edges["node_i"].isin(top_nodes) &
    supra_edges["node_j"].isin(top_nodes)
].copy()

top_edge_file = os.path.join(
    OUT_DIR,
    f"top_component_edges_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

top_edges.to_csv(
    top_edge_file,
    index=False
)

# ============================================================
# 9. PRINT SUMMARY
# ============================================================

print("\n======================================")
print("NODE PARTICIPATION SUMMARY")
print("======================================")

print("\nPhenotype:", PHENOTYPE)
print("Threshold:", P_THRESHOLD)

print("\nNumber of components:")
print(components.shape[0])

print("\nNumber of suprathreshold edges:")
print(supra_edges.shape[0])

print("\nTop component:")
print("Component ID:", top_component["component_id"])
print("Nodes:", top_component["n_nodes"])
print("Edges:", top_component["n_edges"])

if "p_nbs_size" in top_component.index:
    print("NBS p-size:", top_component["p_nbs_size"])

if "p_nbs_mass" in top_component.index:
    print("NBS p-mass:", top_component["p_nbs_mass"])

print("\nTop 20 nodes across all suprathreshold edges:")
print(node_direction.head(20))

print("\nSaved:")
print(out_all_nodes)
print(out_node_direction)
print(out_component_nodes)
print(top_edge_file)

print("\nDONE.")