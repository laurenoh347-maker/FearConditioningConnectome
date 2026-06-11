#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:49:25 2026

@author: bass
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ============================================================
# 0. PATHS
# ============================================================

BASE_DIR = "/Users/bass/LaurenOh/Connectome:Fear Conditioning"

NODE_DIR = os.path.join(BASE_DIR, "NBS_node_participation_results_5-21")

ATLAS_FILE = "/Users/bass/LaurenOh/data/atlas/exvivo_chass/CHASSSYMM3AtlasLegends.xlsx"

PHENOTYPE = "PHENO_context_memory"
P_THRESHOLD = "0.001"

OUT_DIR = os.path.join(BASE_DIR, "NBS_node_participation_labeled")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. INPUT FILES
# ============================================================

node_file = os.path.join(
    NODE_DIR,
    f"node_participation_all_suprathreshold_edges_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

direction_file = os.path.join(
    NODE_DIR,
    f"node_participation_positive_negative_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

component_node_file = os.path.join(
    NODE_DIR,
    f"node_participation_by_component_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

top_edges_file = os.path.join(
    NODE_DIR,
    f"top_component_edges_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

# ============================================================
# 2. LOAD ATLAS
# ============================================================

atlas = pd.read_excel(ATLAS_FILE)

print("Atlas columns:")
print(atlas.columns.tolist())

# Your connectome nodes are 0-based.
# Atlas index is 1-based.
atlas["node_id"] = atlas["index"].astype(int) - 1

keep_cols = [
    "node_id",
    "index",
    "Structure",
    "Abbreviation",
    "Hemisphere",
    "Level_1",
    "Level_2",
    "Level_3",
    "Level_4",
    "Subdivisions_7",
    "Subdivisions_7_nowm"
]

atlas_small = atlas[keep_cols].copy()

# ============================================================
# 3. LABEL NODE PARTICIPATION TABLES
# ============================================================

def label_node_table(infile, outfile):
    df = pd.read_csv(infile)
    df["node_id"] = df["node_id"].astype(int)

    labeled = df.merge(
        atlas_small,
        on="node_id",
        how="left"
    )

    labeled.to_csv(outfile, index=False)

    print("\nSaved:")
    print(outfile)
    print(labeled.head(20))

    return labeled

node_labeled = label_node_table(
    node_file,
    os.path.join(
        OUT_DIR,
        f"LABELED_node_participation_all_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
    )
)

direction_labeled = label_node_table(
    direction_file,
    os.path.join(
        OUT_DIR,
        f"LABELED_node_participation_direction_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
    )
)

component_labeled = label_node_table(
    component_node_file,
    os.path.join(
        OUT_DIR,
        f"LABELED_node_participation_by_component_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
    )
)

# ============================================================
# 4. LABEL TOP COMPONENT EDGES
# ============================================================

edges = pd.read_csv(top_edges_file)

edges["node_i"] = edges["node_i"].astype(int)
edges["node_j"] = edges["node_j"].astype(int)

atlas_i = atlas_small.add_prefix("i_").rename(columns={"i_node_id": "node_i"})
atlas_j = atlas_small.add_prefix("j_").rename(columns={"j_node_id": "node_j"})

edges_labeled = edges.merge(
    atlas_i,
    on="node_i",
    how="left"
)

edges_labeled = edges_labeled.merge(
    atlas_j,
    on="node_j",
    how="left"
)

edges_labeled["edge_label"] = (
    edges_labeled["i_Abbreviation"].astype(str)
    + " - "
    + edges_labeled["j_Abbreviation"].astype(str)
)

edges_labeled["edge_structure"] = (
    edges_labeled["i_Structure"].astype(str)
    + " - "
    + edges_labeled["j_Structure"].astype(str)
)

edges_labeled["direction"] = edges_labeled["t"].apply(
    lambda x: "positive" if x > 0 else "negative"
)

out_edges = os.path.join(
    OUT_DIR,
    f"LABELED_top_component_edges_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

edges_labeled.to_csv(out_edges, index=False)

print("\nSaved labeled top component edges:")
print(out_edges)

print("\nTop labeled edges:")
print(
    edges_labeled[
        [
            "node_i",
            "node_j",
            "edge_label",
            "edge_structure",
            "t",
            "p",
            "direction"
        ]
    ].head(30)
)

# ============================================================
# 5. REGIONAL SUMMARY
# ============================================================

regional_summary = (
    direction_labeled
    .groupby(["Level_1", "Level_2", "Subdivisions_7_nowm"], dropna=False)
    .agg(
        n_nodes=("node_id", "count"),
        total_edges=("n_suprathreshold_edges", "sum"),
        total_positive_edges=("n_positive_edges", "sum"),
        total_negative_edges=("n_negative_edges", "sum")
    )
    .reset_index()
    .sort_values("total_edges", ascending=False)
)

out_regional = os.path.join(
    OUT_DIR,
    f"LABELED_regional_summary_{PHENOTYPE}_pthresh_{P_THRESHOLD}.csv"
)

regional_summary.to_csv(out_regional, index=False)

print("\nSaved regional summary:")
print(out_regional)

print("\nTop regional systems:")
print(regional_summary.head(20))

# ============================================================
# 6. SAVE QUICK RESEARCH SUMMARY
# ============================================================

summary_file = os.path.join(
    OUT_DIR,
    f"LABELED_summary_{PHENOTYPE}_pthresh_{P_THRESHOLD}.txt"
)

with open(summary_file, "w") as f:
    f.write("Labeled NBS node participation summary\n")
    f.write("======================================\n\n")
    f.write(f"Phenotype: {PHENOTYPE}\n")
    f.write(f"Primary threshold: p < {P_THRESHOLD}\n")
    f.write("Atlas mapping: connectome node_id = atlas index - 1\n\n")

    f.write("Top participating nodes:\n")
    f.write(
        direction_labeled[
            [
                "rank",
                "node_id",
                "index",
                "Structure",
                "Abbreviation",
                "Hemisphere",
                "n_suprathreshold_edges",
                "n_positive_edges",
                "n_negative_edges"
            ]
        ].head(20).to_string(index=False)
    )

    f.write("\n\nTop labeled edges:\n")
    f.write(
        edges_labeled[
            [
                "node_i",
                "node_j",
                "edge_label",
                "edge_structure",
                "t",
                "p",
                "direction"
            ]
        ].head(30).to_string(index=False)
    )

    f.write("\n\nTop regional systems:\n")
    f.write(regional_summary.head(20).to_string(index=False))

print("\nSaved summary:")
print(summary_file)

print("\nDONE.")
print("All labeled outputs saved to:")
print(OUT_DIR)