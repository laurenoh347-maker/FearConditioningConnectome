#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import linregress
import os

# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/fearConditioningTrial"

day0_file = os.path.join(BASE_DIR, "FC_Day0_combined_metadata.csv")
day1_file = os.path.join(BASE_DIR, "FC_Day1_combined_metadata.csv")
day2_file = os.path.join(BASE_DIR, "FC_Day2_combined_metadata.csv")

out_file = os.path.join(BASE_DIR, "fear_conditioning_behavior_phenotypes.csv")

FREEZE_COL = "Pct Component Time Freezing"

# ============================================================
# 2. READ DATA
# ============================================================

day0 = pd.read_csv(day0_file)
day1 = pd.read_csv(day1_file)
day2 = pd.read_csv(day2_file)

for df in [day0, day1, day2]:
    df["Animal ID"] = df["Animal ID"].astype(str).str.strip()
    df["Component Name"] = df["Component Name"].astype(str).str.strip()

# ============================================================
# 3. DAY 0: ACQUISITION / LEARNING SLOPE
# ============================================================

sound_map = {
    "Sound 1": 1,
    "Sound 2": 2,
    "Sound 3": 3
}

d0_sound = day0[day0["Component Name"].isin(sound_map.keys())].copy()
d0_sound["trial_num"] = d0_sound["Component Name"].map(sound_map)

def trapezoid_auc_manual(x, y):
    """
    Manual replacement for np.trapz.
    Computes trapezoidal AUC for x,y arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    auc = 0.0
    for i in range(len(x) - 1):
        auc += ((y[i] + y[i + 1]) / 2.0) * (x[i + 1] - x[i])

    return auc

def compute_learning_features(g):
    g = g.sort_values("trial_num")

    x = g["trial_num"].values.astype(float)
    y = g[FREEZE_COL].values.astype(float)

    out = {}

    for t in [1, 2, 3]:
        vals = g.loc[g["trial_num"] == t, FREEZE_COL].values
        out[f"day0_sound{t}_freezing"] = vals[0] if len(vals) > 0 else np.nan

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() >= 2:
        slope, intercept, r, p, se = linregress(x[valid], y[valid])
        out["learning_slope_sound1_to_3"] = slope
        out["learning_intercept"] = intercept
        out["learning_r"] = r
        out["learning_p"] = p
    else:
        out["learning_slope_sound1_to_3"] = np.nan
        out["learning_intercept"] = np.nan
        out["learning_r"] = np.nan
        out["learning_p"] = np.nan

    out["learning_mean_sound1_to_3"] = np.nanmean(y) if np.isfinite(y).any() else np.nan

    # Workaround for missing np.trapz
    out["learning_auc_sound1_to_3"] = trapezoid_auc_manual(x, y)

    out["learning_delta_sound3_minus_sound1"] = (
        out["day0_sound3_freezing"] - out["day0_sound1_freezing"]
    )

    return pd.Series(out)

learning_df = (
    d0_sound
    .groupby("Animal ID")
    .apply(compute_learning_features)
    .reset_index()
)

# ============================================================
# 4. DAY 1: CONTEXTUAL MEMORY
# ============================================================

context_df = day1[day1["Component Name"] == "White House Light"].copy()

context_df = context_df[[
    "Animal ID",
    FREEZE_COL,
    "Genotype",
    "Sex",
    "Lifestyle",
    "Diet",
    "Weight",
    "Age_Months"
]].rename(columns={
    FREEZE_COL: "context_memory_day1_freezing"
})

context_df = (
    context_df
    .groupby("Animal ID")
    .agg({
        "context_memory_day1_freezing": "mean",
        "Genotype": "first",
        "Sex": "first",
        "Lifestyle": "first",
        "Diet": "first",
        "Weight": "first",
        "Age_Months": "first"
    })
    .reset_index()
)

# ============================================================
# 5. DAY 2: CUED / SOUND MEMORY
# ============================================================

d2_wide = day2.pivot_table(
    index="Animal ID",
    columns="Component Name",
    values=FREEZE_COL,
    aggfunc="mean"
).reset_index()

d2_wide = d2_wide.rename(columns={
    "Day2_Acclimation": "day2_acclimation_freezing",
    "Day2_Exploration": "day2_exploration_freezing",
    "Day2_CSTone": "day2_cstone_freezing"
})

d2_wide["cue_memory_delta_cstone_minus_acclimation"] = (
    d2_wide["day2_cstone_freezing"] - d2_wide["day2_acclimation_freezing"]
)

d2_wide["cue_memory_delta_cstone_minus_exploration"] = (
    d2_wide["day2_cstone_freezing"] - d2_wide["day2_exploration_freezing"]
)

# ============================================================
# 6. MERGE ALL BEHAVIOR PHENOTYPES
# ============================================================

behavior = context_df.merge(learning_df, on="Animal ID", how="left")
behavior = behavior.merge(d2_wide, on="Animal ID", how="left")

behavior["baseline_day0_sound1_freezing"] = behavior["day0_sound1_freezing"]
behavior["baseline_day2_acclimation_freezing"] = behavior["day2_acclimation_freezing"]

behavior["PHENO_learning"] = behavior["learning_slope_sound1_to_3"]
behavior["PHENO_context_memory"] = behavior["context_memory_day1_freezing"]
behavior["PHENO_cue_memory"] = behavior["cue_memory_delta_cstone_minus_acclimation"]
behavior["PHENO_cue_memory_vs_exploration"] = behavior["cue_memory_delta_cstone_minus_exploration"]

# ============================================================
# 7. SAVE
# ============================================================

behavior.to_csv(out_file, index=False)

print("Saved phenotype table to:")
print(out_file)

print("\nNumber of animals:", behavior.shape[0])

recommended_cols = [
    "Animal ID",
    "Genotype",
    "Sex",
    "Lifestyle",
    "Diet",
    "Weight",
    "Age_Months",
    "baseline_day0_sound1_freezing",
    "baseline_day2_acclimation_freezing",
    "PHENO_learning",
    "PHENO_context_memory",
    "PHENO_cue_memory",
    "PHENO_cue_memory_vs_exploration"
]

print("\nRecommended columns for connectome mapping:")
print(behavior[recommended_cols].head())

print("\nMissing values in recommended columns:")
print(behavior[recommended_cols].isna().sum())

print("\nPhenotype correlations:")
print(
    behavior[
        [
            "PHENO_learning",
            "PHENO_context_memory",
            "PHENO_cue_memory",
            "baseline_day0_sound1_freezing",
            "baseline_day2_acclimation_freezing"
        ]
    ].corr()
)