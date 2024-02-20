"""
Gather analysis results and form a standard PBEE input file
"""

import os
from glob import glob
import dask.bag as db
import pandas as pd
from osmg.common import G_CONST_IMPERIAL
from src.util import read_study_param

# pylint: disable=invalid-name


def failed_to_converge(logfile):
    """
    Determine if the analysis failed based on the contents of a
    logfile
    """
    with open(logfile, "r", encoding="utf-8") as f:
        contents = f.read()
    return bool("Analysis failed to converge" in contents)


def process_item(item):
    """
    Read all the analysis results and gather the peak results
    considering all ground motion scenarios.

    """

    archetype_code, hz_lvl = item

    space = "3d"

    input_dir = f"results/response/{archetype_code}/individual_files/{hz_lvl}"
    output_dir = f"results/response/{archetype_code}/edp_{space}/{hz_lvl}"

    # determine the number of input files
    # (that should be equal to the number of directories)
    num_inputs = len(glob(f"{input_dir}/*"))

    response_dirs = [
        f"{input_dir}/gm{i+1}"
        for input_dir, i in zip([input_dir] * num_inputs, range(num_inputs))
    ]

    dfs = []
    for i, response_dir in enumerate(response_dirs):
        if space == "2d":
            try:
                df_x = (
                    pd.read_parquet(f"{response_dir}/results_x.parquet")
                    .drop(columns=["time", "Rtime", "Subdiv"])
                    .abs()
                    .max(axis=0)
                )
                fail_x = failed_to_converge(f"{response_dir}/log_x")
                df_y = (
                    pd.read_parquet(f"{response_dir}/results_y.parquet")
                    .drop(columns=["time", "Rtime", "Subdiv"])
                    .abs()
                    .max(axis=0)
                )
                fail_y = failed_to_converge(f"{response_dir}/log_y")
                if (not fail_x) and (not fail_y):
                    df = pd.concat((df_x, df_y)).sort_index()
                    df["FA"] /= G_CONST_IMPERIAL
                    dfs.append(df)
                else:
                    print(f"Warning: {input_dir} failed to converge.")
                    print(f"{response_dir}")
            except FileNotFoundError:
                print(f"Warning: skipping {input_dir}")
                print(f"{response_dir}")
        else:
            try:
                df = (
                    pd.read_parquet(f"{response_dir}/results.parquet")
                    .drop(columns=["time", "Rtime", "Subdiv"])
                    .abs()
                    .max(axis=0)
                    .sort_index()
                )
                df["FA"] /= G_CONST_IMPERIAL
                dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: skipping {response_dir}/results.parquet")

    df_all = pd.concat(dfs, axis=1).T

    # replace column names to highlight the fact that it's peak values

    df_all.columns = df_all.columns.set_levels(
        "P" + df_all.columns.levels[0], level=0
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_all.to_parquet(output_dir + "/response.parquet")


# # Create a Dask client with a specific number of workers/cores
# client = Client(n_workers=12)  # Set the desired number of workers/cores here

# Create a list of items to process
# archetype information
codes = ("smrf", "brbf", "scbf")
stories = ("3", "6", "9")
rcs = ("ii", "iv")
cases = [f"{c}_{s}_{r}" for c in codes for s in stories for r in rcs]
num_hz = int(read_study_param("data/study_vars/m"))
hzs = [f"{x+1}" for x in range(num_hz)]
items = []
for archetype_code in cases:
    for hz in hzs:
        items.append((archetype_code, hz))

# Convert the list to a Dask bag
bag = db.from_sequence(items)

# Apply the function in parallel using Dask's map operation
results = bag.map(process_item)

# Compute the results and retrieve the output
output = results.compute()

print(output)
