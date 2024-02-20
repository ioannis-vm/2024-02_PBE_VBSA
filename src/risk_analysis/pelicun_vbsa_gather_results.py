"""
After running `pelicun_vbsa_loss_realizations.py` on a cluster, this
script gathers the individual results to a single file.
Existing results are updated, new results are appended, missing
results are preserved.
"""

import os
from itertools import product
import glob2
import pandas as pd
from tqdm import tqdm
from src.util import store_info


def locate_files(pattern):
    """
    Find files based on a pattern.
    """
    files = glob2.glob(f"{pattern}")
    files.sort()
    return files


if __name__ == "__main__":

    # loss realizations

    path_items = product(
        ("brbf", "scbf", "smrf"),
        ("3", "6", "9"),
        ("ii", "iv"),
        ("healthcare", "office"),
        ("low", "medium"),
        ("0.8", "1.0", "1.2"),
        ("0.4", "1.0"),
        ("Cost", "Time"),
        [f"{i+1}" for i in range(8)],
    )

    for x in tqdm(list(path_items)):

        paths = [
            f"results/risk/individual_files/{x[0]}_{x[1]}_{x[2]}/"
            f"{x[3]}/{x[8]}/{x[4]}/vbsa_{x[0]}_{x[1]}_{x[2]}_"
            f"{x[3]}_{x[4]}_{x[5]}_{x[6]}_{x[7]}_{x[8]}_{i+1}.parquet"
            for i in range(10)
        ]

        df = pd.concat(
            [
                pd.concat(
                    (pd.read_parquet(path),),
                    keys=pd.MultiIndex.from_tuples((x,)),
                    axis=1,
                )
                for path in paths
            ]
        )

        dfs = []
        for j in range(0, 14):
            dfs.append(df.iloc[:, [0, 1, 2 + (2 * j), 3 + (2 * j)]])
        for sub_df in dfs:
            group = sub_df.columns[-1][-1].replace("D/", "")
            filepath = (
                f"results/risk/individual_files/{x[0]}_{x[1]}_{x[2]}/"
                f"{x[3]}/{x[8]}/{x[4]}/gathered_{x[0]}_{x[1]}_{x[2]}_"
                f"{x[3]}_{x[4]}_{x[5]}_{x[6]}_{x[7]}_{x[8]}_{group}.parquet"
            )
            # sub_df.to_parquet(store_info(filepath))
            sub_df.to_parquet(filepath)

    # replacement probability
    files = locate_files(
        "results/risk/individual_files/*/*/*/*/replacementProb_vbsa_*.txt"
    )

    keys = []
    values = []
    for file in tqdm(files):
        basename = os.path.basename(file).replace(".txt", "")
        split = basename.split("_")
        system = split[2]
        stories = split[3]
        rc = split[4]
        occupancy = split[5]
        modeling_uncertainty = split[6]
        replacement_factor = split[7]
        replacement_threshold = split[8]
        hazard_level = split[9]
        run_iteration = split[10]
        index = split[2:]
        with open(file, "r", encoding="utf-8") as f:
            value = 1.00 - float(f.read())
        keys.append(index)
        values.append(value)

    df_repl = pd.Series(values, index=pd.MultiIndex.from_tuples(keys))
    df_repl.index.names = [
        "system",
        "stories",
        "rc",
        "occupancy",
        "modeling_uncertainty",
        "replacement_factor",
        "replacement_threshold",
        "hazard_level",
        "run_iteration",
    ]
    df_repl = df_repl.unstack().mean(axis=1)
    df_repl = pd.DataFrame(df_repl, columns=["value"])
    df_repl.to_parquet(store_info('results/risk/repl_prob_db.parquet'))
