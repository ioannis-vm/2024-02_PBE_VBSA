"""
Perform VBSA using pelicun
"""

from concurrent.futures import ProcessPoolExecutor
from itertools import product
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from src.util import store_info

pd.options.display.float_format = "{:,.2f}".format

# pylint: disable=protected-access


def calc_sens(
    yA: np.ndarray, yB: np.ndarray, yC: np.ndarray, yD: np.ndarray
) -> tuple[float, float]:
    """
    Calculate variance-based 1st-order and total effect sensitivity
    indices based on the procedure outlined in Saltelli (2002) and
    subsequent improvements discussed in Yun et al. (2017).

    - Saltelli, Andrea. "Making best use of model evaluations to
      compute sensitivity indices." Computer physics communications
      145.2 (2002): 280-297.

    - Yun, Wanying, et al. "An efficient sampling method for
      variance-based sensitivity analysis." Structural Safety 65 (2017):
      74-83.

    Args:
        yA (np.ndarray): One-dimensional numpy array containing realizations
           of model evaluations of analysis 'A'.
        yB (np.ndarray): One-dimensional numpy array containing realizations
           of model evaluations of analysis 'B'
           (every random variable resampled).
        yC (np.ndarray): One-dimensional numpy array containing
           realizations of model evaluations of analysis 'C'. Reusing
           all input realizations of B, except for a single one where
           those of A are used.
        yD (np.ndarray): One-dimensional numpy array containing
           realizations of model evaluations of analysis 'D'. Reusing
           all input realizations of A, except for a single one where
           those of B are used.

    Returns:
        s1, sT: First-order and total effect sensitivity indices

    """

    # # simplest method
    # n = len(yA)
    # f0 = 1./n * np.sum(yA)
    # s1 = ((1./n)*(np.dot(yA, yC)) - f0**2)\
    #     / ((1./n)*(np.dot(yA, yA))-f0**2)
    # sT = 1. - ((1./n)*np.dot(yB, yC) - f0**2)\
    #     / ((1./n)*np.dot(yA, yA) - f0**2)

    # full-use method
    n = len(yA)
    f0_sq = 1.0 / (2.0 * n) * np.sum(yA * yB + yC * yD)
    s1 = ((1 / (2.0 * n)) * (np.sum(yA * yC) + np.sum(yB * yD)) - f0_sq) / (
        (1.0 / (2.0 * n)) * (np.sum(yA**2 + yB**2)) - f0_sq
    )
    sT = 1 - ((1 / (2.0 * n)) * (np.sum(yB * yC) + np.sum(yA * yD)) - f0_sq) / (
        (1.0 / (2.0 * n)) * (np.sum(yA**2 + yB**2)) - f0_sq
    )

    # # Jansen's method
    # n = len(yA)
    # s1 = (1. - ((1./n) * np.sum((yB-yD)**2)) /
    #       ((1./n) * np.sum(yB**2 + yD**2) -
    #        ((1./n) * np.sum(yB))**2 -
    #        ((1./n) * np.sum(yD))**2))
    # sT = (((1./n) * np.sum((yA-yD)**2)) /
    #       ((1./n) * np.sum(yA**2 + yD**2) -
    #        ((1./n) * np.sum(yA))**2 -
    #        ((1./n) * np.sum(yD))**2))

    return s1, sT


# def split_df(df, upper_idx):

#     # split to distinct cases
#     # bring `run_iteration` to the index
#     df = df.stack(level='run_iteration')
#     dfs = []
#     idxs = []
#     df_sub = df
#     cols = df_sub.columns.get_level_values(0).unique()
#     cols = [x for x in cols if x.startswith('C')]
#     for col in cols:
#         rv_group = col.replace('C/', '')
#         df_rvgroup = df_sub[['A', 'B', col, col.replace('C/', 'D/')]]
#         dfs.append(df_rvgroup)
#         idxs.append((*upper_idx, rv_group))

#     return (idxs, dfs)


# # concatenate run cases
# def stack_dfs(dfs):
#     stacked_dfs = []
#     for df in dfs:
#         stacked_dfs.append(df.stack(level=0))
#     return stacked_dfs


def obtain_sis(df):
    """
    Calculate the sensitivity indices
    """

    idx = df.columns[0][:-1]
    rv_group = df.columns[-1][-1].replace("D/", "")
    idx = [*idx, rv_group]
    num_realizations = len(df)

    yA = df.iloc[:, 0].to_numpy().reshape(-1)
    yB = df.iloc[:, 1].to_numpy().reshape(-1)
    yC = df.iloc[:, 2].to_numpy().reshape(-1)
    yD = df.iloc[:, 3].to_numpy().reshape(-1)

    s1, sT = calc_sens(yA, yB, yC, yD)

    # bootstrap
    num_repeats = 1000
    bootstrap_sample_s1 = np.zeros(num_repeats)
    bootstrap_sample_sT = np.zeros(num_repeats)
    for j in range(num_repeats):
        sel = np.random.choice(num_realizations, num_realizations)
        res = calc_sens(yA[sel], yB[sel], yC[sel], yD[sel])
        bootstrap_sample_s1[j] = res[0]
        bootstrap_sample_sT[j] = res[1]
    mean_s1 = bootstrap_sample_s1.mean()
    mean_sT = bootstrap_sample_sT.mean()
    std_s1 = bootstrap_sample_s1.std()
    std_sT = bootstrap_sample_sT.std()
    if np.abs(std_s1) < 1e-10:
        conf_int_s1 = (mean_s1, mean_s1)
    else:
        conf_int_s1 = (
            stats.norm.ppf(0.025, mean_s1, std_s1),
            stats.norm.ppf(0.975, mean_s1, std_s1),
        )
    if np.abs(std_sT) < 1e-10:
        conf_int_sT = (mean_sT, mean_sT)
    else:
        conf_int_sT = (
            stats.norm.ppf(0.025, mean_sT, std_sT),
            stats.norm.ppf(0.975, mean_sT, std_sT),
        )

    sens_results_df = pd.DataFrame(
        {
            "s1": s1,
            "sT": sT,
            "s1_CI_l": conf_int_s1[0],
            "s1_CI_h": conf_int_s1[1],
            "sT_CI_l": conf_int_sT[0],
            "sT_CI_h": conf_int_sT[1],
        },
        index=pd.MultiIndex.from_tuples((idx,)),
    )
    sens_results_df.columns.name = "Case"

    return sens_results_df


def process_path(path_item):
    """
    Get sensitivity indices from loss estimation files
    """
    x = path_item
    filepath = (
        f"results/risk/individual_files/{x[0]}_{x[1]}_{x[2]}/"
        f"{x[3]}/{x[8]}/{x[4]}/gathered_{x[0]}_{x[1]}_{x[2]}_"
        f"{x[3]}_{x[4]}_{x[5]}_{x[6]}_{x[7]}_{x[8]}_{x[9]}.parquet"
    )
    sub_df = pd.read_parquet(filepath)
    si_df = obtain_sis(sub_df)
    return si_df


if __name__ == "__main__":

    path_items = list(
        product(
            ("brbf", "scbf", "smrf"),
            ("3", "6", "9"),
            ("ii", "iv"),
            ("healthcare", "office"),
            ("low", "medium"),
            ("0.8", "1.0", "1.2"),
            ("0.4", "1.0"),
            ("Cost", "Time"),
            [f"{i+1}" for i in range(8)],
            (
                "EDP",
                "EDP-PID",
                "EDP-PFV",
                "EDP-PFA",
                "EDP-RID",
                "CMP",
                "C-DS",
                "C-DS-FRG",
                "C-DS-LS",
                "B-DS",
                "B-DSc",
                "B-DSe",
                "C-DV",
                "B-DV",
            ),
        )
    )

    path_items = path_items[0:10]

    with ProcessPoolExecutor() as executor:
        si_dfs = list(
            tqdm(executor.map(process_path, path_items), total=len(path_items))
        )

    # si_dfs = []
    # for path_item in tqdm(list(path_items)[0:20]):
    #     si_df = process_path(path_item)
    #     si_dfs.append(si_df)

    si_df_all = pd.concat(si_dfs)
    si_df_all.sort_index(inplace=True)

    si_df_all.to_parquet(store_info("results/risk/si_db.parquet"))
