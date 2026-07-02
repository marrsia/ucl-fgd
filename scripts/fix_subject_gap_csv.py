"""
Fix wilcox2022_embed4_subject_gap.csv: subject column is currently filled for
gap=yes rows and empty for gap=no rows — the opposite of what it should be.
This script swaps subject values within each matched pair.
"""

import pandas as pd

PATH = "data/stimuli/wilcox2022_embed4_subject_gap.csv"

df = pd.read_csv(PATH)

group_keys = ["sentence_group", "wh", "levels_of_embedding"]

for _, grp in df.groupby(group_keys):
    gap_idx    = grp[grp["gap"] == "yes"].index
    no_gap_idx = grp[grp["gap"] == "no"].index

    for g, ng in zip(gap_idx, no_gap_idx):
        subject = df.at[g, "subject"]
        df.at[ng, "subject"] = subject
        df.at[g,  "subject"] = float("nan")

df.to_csv(PATH, index=False)

# Verify
gap_empty         = df[df["gap"] == "yes"]["subject"].isna().sum()
no_gap_has_subj   = df[df["gap"] == "no"]["subject"].notna().sum()
print(f"gap=yes rows with empty subject:  {gap_empty}  (expected 540)")
print(f"gap=no  rows with subject filled: {no_gap_has_subj}  (expected 540)")
