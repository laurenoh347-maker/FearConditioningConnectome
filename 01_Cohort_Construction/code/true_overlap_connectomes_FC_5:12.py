#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:21:08 2026

@author: bass
"""
import os
import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path("/Users/bass/LaurenOh/data")

FC_CSV = DATA_DIR / "/Users/bass/LaurenOh/Connectome:Fear Conditioning/fearConditioningTrial/FC_Day1_combined_metadata.csv"
CONNECTOME_DIR = DATA_DIR / "/Users/bass/LaurenOh/data/connectomes"

fc_df = pd.read_csv(FC_CSV, dtype=str)
fc_df.columns = fc_df.columns.str.strip()
fc_df["Animal ID"] = fc_df["Animal ID"].astype(str).str.strip()

records = []

for path in sorted(CONNECTOME_DIR.glob("FC_A*.csv")):
    match = re.search(r"A(\d{8})", path.name)
    if match is None:
        continue

    digits = match.group(1)
    animal_id = f"{digits[:6]}_{int(digits[6:])}"

    records.append({
        "Animal ID": animal_id,
        "Connectome_File": path.name
    })

conn_df = pd.DataFrame(records)

overlap_df = fc_df.merge(conn_df, on="Animal ID", how="inner")

out_csv = DATA_DIR / "true_overlap_between_connectome_and_fear_conditioning_5-12.csv"
overlap_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Overlap animals:", overlap_df["Animal ID"].nunique())
print(overlap_df[["Animal ID", "Connectome_File"]])