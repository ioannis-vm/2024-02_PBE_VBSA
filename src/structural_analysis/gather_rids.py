"""
Determine residual drifts
"""

from itertools import product
import pandas as pd
from tqdm import tqdm

types = ("smrf", "scbf", "brbf")
stors = ("3", "6", "9")
rcs = ("ii", "iv")
hzs = [f"{i+1}" for i in range(8)]
gms = [f"gm{i+1}" for i in range(40)]

keys = []
dfs = []

total = len(types) * len(stors) * len(rcs) * len(hzs)
pbar = tqdm(total=total, unit="item")
for tp, st, rc, hz in product(types, stors, rcs, hzs):

    pbar.update(1)

    archetype = f"{tp}_{st}_{rc}"

    summary_df_path_updated = (
        f"results/response/{archetype}/edp_3d/{hz}/response_rid.parquet"
    )

    df = pd.read_parquet(summary_df_path_updated)
    keys.append((tp, st, rc, hz))
    dfs.append(df)


pbar.close()

df = pd.concat(dfs, axis=1, keys=keys)
df.index.names = ("gm",)
df.columns.names = ("system", "stories", "rc", "hz", "edp", "loc", "dir")
df = pd.DataFrame(df.T.stack(), columns=["value"])
df.to_parquet("edp.parquet")
