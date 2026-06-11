#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/fearConditioningTrial"

NBS_DIR = os.path.join(
    BASE_DIR,
    "NBS_overlap_results_PThreshIs0Point001_6-4",
    "PHENO_context_memory"
)

component_edges_file = os.path.join(
    NBS_DIR,
    "component_1_edges.csv"
)

ATLAS_DIR = "/Users/bass/LaurenOh/data/atlas/exvivo_chass"

# ============================================================
# FIND ATLAS FILE AUTOMATICALLY
# ============================================================

excel_files = glob.glob(os.path.join(ATLAS_DIR, "*.xlsx"))

if len(excel_files) == 0:
    raise FileNotFoundError(
        f"No Excel atlas file found in:\n{ATLAS_DIR}"
    )

atlas_file = excel_files[0]

print("Using atlas:")
print(atlas_file)

# ============================================================
# OUTPUT FILES
# ============================================================

out_file = os.path.join(
    NBS_DIR,
    "component_1_edges_annotated.csv"
)

node_summary_file = os.path.join(
    NBS_DIR,
    "component_1_node_summary.csv"
)

network_summary_file = os.path.join(
    NBS_DIR,
    "component_1_network_level_summary.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading files...")

edges = pd.read_csv(component_edges_file)
atlas = pd.read_excel(atlas_file)

print("Edges shape:", edges.shape)
print("Atlas shape:", atlas.shape)

print("\nAtlas columns:")
print(atlas.columns.tolist())

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_cols = [
    "index",
    "Structure",
    "Abbreviation",
    "Hemisphere"
]

missing = [c for c in required_cols if c not in atlas.columns]

if len(missing) > 0:
    raise ValueError(
        "Atlas missing columns:\n" + str(missing)
    )

# ============================================================
# CLEAN ATLAS
# ============================================================

atlas = atlas.copy()

atlas["index"] = pd.to_numeric(
    atlas["index"],
    errors="coerce"
)

atlas = atlas.dropna(subset=["index"])

atlas["index"] = atlas["index"].astype(int)

# convert atlas numbering to connectome numbering
atlas["node_0based"] = atlas["index"] - 1

atlas = atlas[
    (atlas["node_0based"] >= 0) &
    (atlas["node_0based"] < 324)
]

print("\nAtlas rows retained:", atlas.shape[0])

# ============================================================
# MERGE NODE I
# ============================================================

annot = edges.merge(
    atlas.add_suffix("_i"),
    left_on="node_i",
    right_on="node_0based_i",
    how="left"
)

# ============================================================
# MERGE NODE J
# ============================================================

annot = annot.merge(
    atlas.add_suffix("_j"),
    left_on="node_j",
    right_on="node_0based_j",
    how="left"
)

# ============================================================
# EDGE LABELS
# ============================================================

annot["edge_label"] = (
    annot["Structure_i"].astype(str)
    + " ("
    + annot["Hemisphere_i"].astype(str)
    + ") -- "
    + annot["Structure_j"].astype(str)
    + " ("
    + annot["Hemisphere_j"].astype(str)
    + ")"
)

annot["abs_t"] = np.abs(annot["t_value"])

annot = annot.sort_values(
    "abs_t",
    ascending=False
)

annot.to_csv(out_file, index=False)

print("\nSaved:")
print(out_file)

# ============================================================
# NODE SUMMARY
# ============================================================

nodes_i = annot[[
    "node_i",
    "Structure_i",
    "Hemisphere_i"
]].rename(columns={
    "node_i": "node",
    "Structure_i": "Structure",
    "Hemisphere_i": "Hemisphere"
})

nodes_j = annot[[
    "node_j",
    "Structure_j",
    "Hemisphere_j"
]].rename(columns={
    "node_j": "node",
    "Structure_j": "Structure",
    "Hemisphere_j": "Hemisphere"
})

node_table = pd.concat(
    [nodes_i, nodes_j],
    axis=0
)

node_summary = (
    node_table
    .groupby(
        ["node", "Structure", "Hemisphere"]
    )
    .size()
    .reset_index(name="degree")
    .sort_values(
        "degree",
        ascending=False
    )
)

node_summary.to_csv(
    node_summary_file,
    index=False
)

print("\nSaved:")
print(node_summary_file)

# ============================================================
# TOP EDGES
# ============================================================

print("\n==============================")
print("TOP EDGES")
print("==============================")

print(
    annot[
        [
            "node_i",
            "node_j",
            "edge_label",
            "t_value"
        ]
    ].head(20)
)

# ============================================================
# TOP NODES
# ============================================================

print("\n==============================")
print("TOP NODES")
print("==============================")

print(
    node_summary.head(20)
)

# ============================================================
# SIMPLE NETWORK SUMMARY
# ============================================================

network_summary = (
    node_table
    .groupby("Structure")
    .size()
    .reset_index(name="occurrences")
    .sort_values(
        "occurrences",
        ascending=False
    )
)

network_summary.to_csv(
    network_summary_file,
    index=False
)

print("\nSaved:")
print(network_summary_file)

print("\nDONE")