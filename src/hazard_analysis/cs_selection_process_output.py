"""
Generates an input file for use with CS_Selection in order to select
ground motions for each archetype and hazard level.

"""

from math import ceil
import numpy as np
import pandas as pd
from src.util import read_study_param

# initialize
dataframe_rows = []

# archetype information
codes = ("smrf", "scbf", "brbf")
stories = ("3", "6", "9")
rcs = ("ii", "iv")
cases = [f"{c}_{s}_{r}" for c in codes for s in stories for r in rcs]

num_hz = int(read_study_param("data/study_vars/m"))
ngm = int(read_study_param("data/study_vars/ngm"))

# initialize
dfs_arch = []
conditioning_periods = pd.Series(np.empty(len(cases)), index=cases)

for arch in cases:
    t_bar = float(read_study_param(f"data/{arch}/period"))
    conditioning_periods[arch] = t_bar

    # initialize
    dfs_hz = []
    for hz in range(num_hz):
        path = f"results/site_hazard/{arch}/required_records_hz_{hz+1}.txt"
        df = pd.read_csv(path, skiprows=6, sep="	", index_col=0, header=[0])
        df.columns = [x.strip() for x in df.columns]
        df = df.loc[:, ("Record Sequence Number", "Scale Factor")]
        df = df.sort_values(by="Record Sequence Number")
        df.index = range(1, ngm + 1)
        df.index.name = "Record Number"
        df.columns = ["RSN", "SF"]
        dfs_hz.append(df)
    df = pd.concat(dfs_hz, axis=1, keys=[f"hz_{i+1}" for i in range(num_hz)])
    dfs_arch.append(df)

df = pd.concat(dfs_arch, axis=1, keys=cases)
df.columns.names = ["archetype", "hazard_level", "quantity"]
df = df.T

# store deaggregation results for all achetypes in the form of a csv
# file
df.to_csv("results/site_hazard/required_records_and_scaling_factors.csv")

# dfrsn = df.xs('RSN', level=2).astype(int)
# for col in dfrsn.columns:
#     for row in dfrsn[col]:
#         if row == 465:
#             print(row, dfrsn[col][dfrsn[col] == 465])
# for col in dfrsn.columns:
#     for row in dfrsn[col]:
#         if row == 498:
#             print(row, dfrsn[col][dfrsn[col] == 498])

# obtain unique RSNs to download from the ground motion database
rsns = df.xs("RSN", level=2).astype(int).unstack().unstack().unique()
rsns = pd.Series(rsns).sort_values()
rsns.index = range(len(rsns))
# num_times = (df.xs('RSN', level=2).astype(int)
#              .unstack().unstack().value_counts())

gm_group = pd.Series(index=rsns, dtype="int")
num_groups = ceil(len(rsns) / 100)
for group in range(num_groups):
    istart = 100 * group
    iend = min(100 + 100 * group, len(rsns))
    gm_group[rsns.iloc[istart:iend]] = group
    with open(
        f"results/site_hazard/rsns_unique_{group+1}.txt", "w", encoding="utf-8"
    ) as f:
        f.write(", ".join([f"{r}" for r in rsns.iloc[istart:iend]]))
gm_group = gm_group.astype(int)
gm_group.index.name = "RSN"
gm_group.name = "group"
gm_group.to_csv("results/site_hazard/ground_motion_group.csv")
