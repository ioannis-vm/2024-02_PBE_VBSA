"""
Use analysis results to derive the collapse fragility of each
archetype
"""

import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from src.util import read_study_param
from src.util import store_info


def process_response(filepath):
    """
    Process the response file
    """
    df = pd.read_parquet(filepath)
    df.columns.names = ("edp", "location", "direction")

    num_runs = len(df)

    drift_threshold = 0.06

    collapse = df["PID"].max(axis=1) > drift_threshold

    collapse_idx = collapse[collapse == True].index
    num_collapse = len(collapse_idx)

    return num_collapse, num_runs


def get_sa(hz):
    """
    Read a target spectrum from a file.
    """
    # determine Sa at those levels
    spectrum = pd.read_csv(f"results/site_hazard/{hz}.csv", index_col=0, header=0)

    ifun = interp1d(spectrum.index.to_numpy(), spectrum.to_numpy().reshape(-1))
    current_sa = float(ifun(base_period))
    return current_sa


def neg_log_likelihood(x, njs, zjs, xjs):
    """
    Calculates the negative log likelihood of observing the given data
    under the specified distribution parameters
    """
    theta, beta = x
    phi = norm.cdf(np.log(xjs / theta) / beta)
    logl = np.sum(
        np.log(binom(njs, zjs))
        + zjs * np.log(phi)
        + (njs - zjs) * np.log(1.00 - phi)
    )
    return -logl


# archetype information
codes = ("smrf", "scbf", "brbf")
stories = ("3", "6", "9")
rcs = ("ii", "iv")
cases = [f"{c}_{s}_{r}" for r in rcs for c in codes for s in stories]

num_hz = int(read_study_param("data/study_vars/m"))

results = {}

for archetype in cases:

    base_period = float(read_study_param(f"data/{archetype}/period"))

    filepaths = [
        f"results/response/{archetype}/edp/{x+1}/response.parquet"
        for x in range(num_hz)
    ]

    zjs = []
    njs = []
    for filepath in filepaths:
        z, n = process_response(filepath)
        zjs.append(z)
        njs.append(n)
    xjs = []
    for hz in [f"UHS_{i}" for i in range(1, 9)]:
        xjs.append(get_sa(hz))

    zjs = np.array(zjs, dtype=float)
    njs = np.array(njs, dtype=float)
    xjs = np.array(xjs, dtype=float)

    # assign a baseline of a 0.025% collapse probability in cases
    # where collapse is not observed
    if all(zjs == 0.00):
        zjs[7] = 0.05 * 40.00
        zjs[6] = 0.025 * 40.00

    x0 = np.array((3.00, 0.40))

    res = minimize(
        neg_log_likelihood,
        x0,
        method="nelder-mead",
        args=(njs, zjs, xjs),
        bounds=((0.0, 20.00), (0.40, 0.40)),
    )

    results[archetype] = {
        "zjs": zjs,
        "njs": njs,
        "xjs": xjs,
        "median": res.x[0],
        "beta": res.x[1],
    }

# dataframe with prob of collapse for each hazard level
df_quantiles = pd.DataFrame(
    [results[archetype]["zjs"] / results[archetype]["njs"] for archetype in cases],
    index=[x.upper().replace("_", " ") for x in cases],
    columns=range(1, num_hz + 1),
)
df_quantiles.columns.name = "Hazard Level"
df_quantiles.index.name = "Archetype"
# df_quantiles.sort_values(by=[8], ascending=False, inplace=True)
print(df_quantiles)
print()

# dataframe with fitted fragility curves
df_fragility = pd.DataFrame.from_dict(
    {
        "Median": [results[archetype]["median"] for archetype in cases],
        "Beta": [results[archetype]["beta"] for archetype in cases],
    },
)
df_fragility.index = [x.upper().replace("_", " ") for x in cases]
df_fragility.index.name = "Archetype"
print(df_fragility)
print()

df_fragility.to_csv(store_info("results/response/collapse_fragilities.csv"))
