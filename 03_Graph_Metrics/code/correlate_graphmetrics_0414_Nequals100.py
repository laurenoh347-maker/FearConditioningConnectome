#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:02:19 2026

@author: bass
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------

clustering_vals = np.load("/Users/bass/LaurenOh/results/clustering_vals.npy")
path_lengths = np.load("/Users/bass/LaurenOh/results/path_lengths.npy")

# --------------------------------------------------------
# CLEAN
# --------------------------------------------------------

mask = np.isfinite(clustering_vals) & np.isfinite(path_lengths)

clustering_clean = clustering_vals[mask]
path_clean = path_lengths[mask]

print("Valid subjects:", len(clustering_clean))

# --------------------------------------------------------
# CORRELATION TESTS
# --------------------------------------------------------

r_p, p_p = pearsonr(clustering_clean, path_clean)
r_s, p_s = spearmanr(clustering_clean, path_clean)

print("\n=== ASSOCIATION TEST ===")
print(f"Pearson r = {r_p:.3f}, p = {p_p:.5f}")
print(f"Spearman r = {r_s:.3f}, p = {p_s:.5f}")

# --------------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------------

plt.figure()
plt.scatter(clustering_clean, path_clean, alpha=0.7)

# regression line
m, b = np.polyfit(clustering_clean, path_clean, 1)
plt.plot(clustering_clean, m*clustering_clean + b)

plt.xlabel("Clustering Coefficient")
plt.ylabel("Path Length")
plt.title("Clustering vs Path Length")
plt.show()