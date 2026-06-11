#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:51:26 2026

@author: bass
"""

import pandas as pd

# -------------------------
# FILES
# -------------------------

meta_file = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/qial_metadata_appended_040126.xlsx"

fc_day0 = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/fearConditioningTrial/FC_Day0_combined_metadata.csv"
fc_day1 = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/fearConditioningTrial/FC_Day1_combined_metadata.csv"
fc_day2 = "/Users/bass/LaurenOh/Connectome:Fear Conditioning/fearConditioningTrial/FC_Day2_combined_metadata.csv"

# -------------------------
# LOAD DATA
# -------------------------

meta = pd.read_excel(meta_file)

d0 = pd.read_csv(fc_day0)
d1 = pd.read_csv(fc_day1)
d2 = pd.read_csv(fc_day2)

# -------------------------
# CLEANING FUNCTION
# -------------------------

def clean(x):
    x = str(x).strip()

    # remove Excel float artifacts
    if "." in x:
        x = x.split(".")[0]

    return x

# -------------------------
# STANDARDIZE IDS
# -------------------------

meta['BadeaID'] = meta['BadeaID'].apply(clean)
meta['ARunno'] = meta['ARunno'].apply(clean)

# adjust if needed
meta['DWI'] = meta['DWI'].astype(str).apply(clean)

for df in [d0, d1, d2]:
    df['Animal ID'] = df['Animal ID'].apply(clean)

# -------------------------
# COMBINE FC DATA
# -------------------------

fc_all = pd.concat([d0, d1, d2], ignore_index=True)

# unique animals only
fc_animals = fc_all[['Animal ID']].drop_duplicates()

# -------------------------
# MERGE: FC → META
# -------------------------

merged = fc_animals.merge(
    meta,
    left_on='Animal ID',
    right_on='BadeaID',
    how='left'
)

# -------------------------
# SELECT RELEVANT COLUMNS
# -------------------------

result = merged[['BadeaID', 'Animal ID', 'ARunno', 'DWI']]

# -------------------------
# QUALITY CHECKS
# -------------------------

print("\n=== SUMMARY ===")
print(f"FC animals: {len(fc_animals)}")
print(f"Matched in metadata: {result['BadeaID'].notna().sum()}")
print(f"Missing mappings: {result['BadeaID'].isna().sum()}")

print("\nExamples of missing:")
print(result[result['BadeaID'].isna()].head())

# -------------------------
# SAVE OUTPUT
# -------------------------

output_file = "FC_to_Imaging_Mapping.csv"
result.to_csv(output_file, index=False)

print("\nSaved:", output_file)