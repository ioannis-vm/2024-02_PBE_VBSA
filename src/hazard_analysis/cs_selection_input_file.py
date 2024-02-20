"""
Generates an input file for use with CS_Selection in order to select
ground motions for each archetype and hazard level.

"""

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
vs30 = float(read_study_param("data/study_vars/vs30"))

# initialize
dfs_arch = []
conditioning_periods = pd.Series(np.empty(len(cases)), index=cases)

for arch in cases:
    t_bar = float(read_study_param(f"data/{arch}/period_closest"))
    conditioning_periods[arch] = t_bar

    # initialize
    dfs_hz = []
    for hz in range(num_hz):
        path = f"results/site_hazard/{arch}/deaggregation_{hz+1}.txt"
        df = pd.read_csv(
            path,
            skiprows=2,
            skipfooter=4,
            sep=" = ",
            index_col=0,
            engine="python",
            header=None,
        )
        df.index.name = "parameter"
        df.columns = [f"hz_{hz+1}"]
        dfs_hz.append(df)
    df = pd.concat(dfs_hz, axis=1)
    dfs_arch.append(df)

df = pd.concat(dfs_arch, axis=1, keys=cases)
df.columns.names = ["archetype", "hazard_level"]
df = df.T

# store deaggregation results for all achetypes in the form of a csv
# file
df.to_csv("results/site_hazard/deaggregation.csv")

# generate input file for CS_Selection
rows = []
for arch in cases:
    for hz in range(num_hz):
        rows.append(
            [
                conditioning_periods[arch],
                df.at[(arch, f"hz_{hz+1}"), "Mbar"],
                df.at[(arch, f"hz_{hz+1}"), "Dbar"],
                df.at[(arch, f"hz_{hz+1}"), "Ebar"],
                vs30,
                f"results/site_hazard/{arch}/",
                f"required_records_hz_{hz+1}.txt",
                arch,
            ]
        )
df_css = pd.DataFrame(
    rows,
    columns=[
        "Tcond",
        "M_bar",
        "Rjb",
        "eps_bar",
        "Vs30",
        "outputDir",
        "outputFile",
        "code",
    ],
)
df_css.to_csv("results/site_hazard/CS_Selection_input_file.csv")
