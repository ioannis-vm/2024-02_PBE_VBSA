"""
  Get the maximum scaling factor coming out of CS_Selection.
"""

import pandas as pd

df = pd.read_csv(
    "results/site_hazard/required_records_and_scaling_factors_adjusted_to_cms.csv",
    index_col=[0, 1, 2],
)

df_sf = df.xs("SF", level=2, axis=0)
max_scaling_factor = df_sf.max(axis=1).max()

print(max_scaling_factor)
