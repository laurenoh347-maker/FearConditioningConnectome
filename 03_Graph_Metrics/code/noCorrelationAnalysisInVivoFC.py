#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:24:54 2026

@author: bass
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------
# USER SETTINGS
# --------------------------------------------------------

BEHAVIOR_CSV = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/connectomeFCDataResults/FC_standardized_behavior.csv"
CONNECTOME_DIR = "/Users/bass/LaurenOh/data/connectomes"
OUT_DIR = "/Users/bass/LaurenOh/FearConditioning/results/amy_hipp_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

behavior_cols = ["Day0_Slope", "Day1_Context", "Day2_Cued"]

# --------------------------------------------------------
# ROI indices (FINAL CORRECT VERSION)
# --------------------------------------------------------

LEFT_AMY = 64
LEFT_HIPP = 50

RIGHT_AMY = 230
RIGHT_HIPP = 216

# --------------------------------------------------------
# Helper: extract animal ID
# --------------------------------------------------------

def extract_animal_id(fname):
    m = re.search(r"A(\d{8})", os.path.basename(fname))
    if not m:
        return None
    digits = m.group(1)
    date = digits[:6]
    animal_num = str(int(digits[6:]))  # removes leading zero
    return f"{date}_{animal_num}"

# --------------------------------------------------------
# Load behavior
# --------------------------------------------------------

beh = pd.read_csv(BEHAVIOR_CSV)
beh["Animal ID"] = beh["Animal ID"].astype(str).str.strip()
beh = beh.set_index("Animal ID")

print("Behavior subjects:", len(beh))

# --------------------------------------------------------
# Load connectomes
# --------------------------------------------------------

files = sorted(glob.glob(os.path.join(CONNECTOME_DIR, "*.csv")))
print("Total connectome files found:", len(files))

connectivity = []
animal_ids = []

for f in files:

    aid = extract_animal_id(f)
    if aid is None:
        continue

    try:
        mat = pd.read_csv(f, index_col=0).to_numpy()

        if mat.shape != (323, 323):
            continue

        # Extract connectivity
        left_conn = mat[LEFT_AMY, LEFT_HIPP]
        right_conn = mat[RIGHT_AMY, RIGHT_HIPP]

        mean_conn = np.mean([left_conn, right_conn])

        connectivity.append(mean_conn)
        animal_ids.append(aid)

    except Exception as e:
        print("Skipping:", os.path.basename(f), e)

connectivity = np.array(connectivity)
animal_ids = np.array(animal_ids)

print("Loaded connectomes:", len(connectivity))

# --------------------------------------------------------
# Align datasets
# --------------------------------------------------------

animal_ids = np.array([str(a).strip() for a in animal_ids])
beh.index = beh.index.astype(str).str.strip()

common_ids = np.intersect1d(animal_ids, beh.index.values)

print("Common subjects:", len(common_ids))

if len(common_ids) == 0:
    raise ValueError("No overlapping animal IDs found")

# Align
conn_dict = dict(zip(animal_ids, connectivity))
conn_aligned = np.array([conn_dict[i] for i in common_ids])

df = beh.loc[common_ids].copy()
df["AmyHipp_Connectivity"] = conn_aligned

print("Final dataset size:", df.shape)

# --------------------------------------------------------
# Fisher z-transform
# --------------------------------------------------------

df["AmyHipp_z"] = np.arctanh(df["AmyHipp_Connectivity"])

# --------------------------------------------------------
# Regression
# --------------------------------------------------------

results = []

for behavior in behavior_cols:

    y = df[behavior].values
    x = df["AmyHipp_z"].values

    slope, intercept, r, p, se = stats.linregress(x, y)

    results.append({
        "Behavior": behavior,
        "Beta": slope,
        "r": r,
        "p_uncorrected": p
    })

# --------------------------------------------------------
# FDR correction (3 tests)
# --------------------------------------------------------

pvals = np.array([r["p_uncorrected"] for r in results])
order = np.argsort(pvals)
ranked = pvals[order]

adj = ranked * len(pvals) / (np.arange(len(pvals)) + 1)
adj = np.minimum.accumulate(adj[::-1])[::-1]

p_fdr = np.empty_like(adj)
p_fdr[order] = adj

for i in range(len(results)):
    results[i]["p_FDR"] = p_fdr[i]

results_df = pd.DataFrame(results)

# Save
results_df.to_csv(os.path.join(OUT_DIR, "AmyHipp_results.csv"), index=False)

# --------------------------------------------------------
# Output
# --------------------------------------------------------

print("\nRESULTS:")
print(results_df)

print("\nAnalysis complete.")