"""
Compare the 3D analysis results to the 2D analysis results.
"""

from itertools import product
import os
import numpy as np
import pandas as pd


# We want to load all analysis results and compare the EDPs against each other.
systems = ("smrf", "scbf", "brbf")
stories = ("3", "6", "9")
rcs = ("ii", "iv")
hzs = [f"{i+1}" for i in range(8)]
gms = [f"gm{i+1}" for i in range(40)]


def get_max_subdiv(time_vec):
    """
    Get the largest time-step subdivision level
    """
    dt = time_vec[1] - time_vec[0]
    diff = np.round(-np.log10(np.diff(time_vec) / dt))
    diff = np.array((0.00, *diff))
    return np.max(diff)


items = []
vals = []

total = len(systems) * len(stories) * len(rcs) * len(hzs) * len(gms)

for syst, stor, rc, hz, gm in product(systems, stories, rcs, hzs, gms):
    idx = (syst, stor, rc, hz, gm)
    items.append(idx)


def process_item(item):
    syst, stor, rc, hz, gm = item
    archetype = f"{syst}_{stor}_{rc}"
    response_3d_path = (
        f"results/response/{archetype}/individual_files/{hz}/{gm}/results.parquet"
    )
    if os.path.exists(response_3d_path):
        data_3d = pd.read_parquet(response_3d_path)
        max_subdiv_3d = get_max_subdiv(data_3d["time"].to_numpy().reshape(-1))
    else:
        max_subdiv_3d = np.nan
    response_2d_x_path = (
        f"results/response/{archetype}/individual_files_2d/{hz}/{gm}/results_x.parquet"
    )
    if os.path.exists(response_2d_x_path):
        data_x = pd.read_parquet(response_2d_x_path)
        max_subdiv_2dx = get_max_subdiv(data_x["time"].to_numpy().reshape(-1))
    else:
        max_subdiv_2dx = np.nan
    response_2d_y_path = (
        f"results/response/{archetype}/individual_files_2d/{hz}/{gm}/results_y.parquet"
    )
    if os.path.exists(response_2d_y_path):
        data_y = pd.read_parquet(response_2d_y_path)
        if "time" in data_y.columns.get_level_values(0):
            max_subdiv_2dy = get_max_subdiv(data_y["time"].to_numpy().reshape(-1))
        else:
            max_subdiv_2dy = np.nan
    else:
        max_subdiv_2dy = np.nan
    return max_subdiv_3d, max_subdiv_2dx, max_subdiv_2dy


df = pd.read_parquet("results/response/subdiv_stats.parquet")

# group all gms and use the peak
print(df.groupby(by=["arch", "stor", "hz"], axis=0).max().to_string())

print(df.loc[("smrf", "3", "ii"), :].to_string())
