#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------------------------------------
# SETTINGS
# --------------------------------------------------------

CONNECTOME_DIR = "/Users/bass/LaurenOh/data/connectomes"

# Use subset for speed (change later)
MAX_FILES = 100

# --------------------------------------------------------
# Helper
# --------------------------------------------------------

def load_matrix(f):
    df = pd.read_csv(f)

    # keep only numeric
    df = df.select_dtypes(include=[np.number])
    mat = df.to_numpy()

    # fix off-by-one issues
    if mat.shape[1] == mat.shape[0] + 1:
        mat = mat[:, :-1]
    elif mat.shape[0] == mat.shape[1] + 1:
        mat = mat[:-1, :]

    return mat

# --------------------------------------------------------
# Load files
# --------------------------------------------------------

files = sorted(glob.glob(os.path.join(CONNECTOME_DIR, "*.csv")))
files = files[:MAX_FILES]

print("Processing files:", len(files))

clustering_vals = []
path_lengths = []

# --------------------------------------------------------
# Main loop
# --------------------------------------------------------

for i, f in enumerate(files):

    try:
        mat = load_matrix(f)

        if mat.shape != (323, 323):
            continue

        if not np.all(np.isfinite(mat)):
            continue

        # --- weights & distance ---
        weights = np.abs(mat)
        weights[weights == 0] = 1e-6
        dist = 1 / weights

        # Graphs
        G_dist = nx.from_numpy_array(dist)
        G_weight = nx.from_numpy_array(weights)

        # Largest component
        largest_cc = max(nx.connected_components(G_dist), key=len)
        G_dist = G_dist.subgraph(largest_cc).copy()

        # Metrics
        clustering = nx.average_clustering(G_weight, weight="weight")

        try:
            path_len = nx.average_shortest_path_length(G_dist, weight="weight")
        except:
            path_len = np.nan

        clustering_vals.append(clustering)
        path_lengths.append(path_len)

        if i % 10 == 0:
            print(f"Processed {i} files")

    except Exception as e:
        print("Skipping:", os.path.basename(f), e)

# --------------------------------------------------------
# Convert
# --------------------------------------------------------

clustering_vals = np.array(clustering_vals)
path_lengths = np.array(path_lengths)

print("\n=== RESULTS ===")
print("Subjects:", len(clustering_vals))
print("Clustering (first 5):", clustering_vals[:5])
print("Path length (first 5):", path_lengths[:5])

# --------------------------------------------------------
# Plot distributions
# --------------------------------------------------------

plt.figure()
plt.hist(clustering_vals, bins=20)
plt.title("Clustering Distribution")
plt.xlabel("Clustering")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.hist(path_lengths[~np.isnan(path_lengths)], bins=20)
plt.title("Path Length Distribution")
plt.xlabel("Path Length")
plt.ylabel("Count")
plt.show()

np.save("/Users/bass/LaurenOh/results/clustering_vals.npy", clustering_vals)
np.save("/Users/bass/LaurenOh/results/path_lengths.npy", path_lengths)