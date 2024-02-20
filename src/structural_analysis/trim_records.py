"""
Manually inspect and trim ground motion records if required.
"""

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from osmg.ground_motion_utils import import_PEER
from src.util import retrieve_peer_gm_data

df_records = pd.read_csv(
    "results/site_hazard/required_records_and_scaling_factors_adjusted_to_cms.csv",
    index_col=[0, 1, 2],
)
# get unique RSNs
uniq_rsns = set(
    df_records.xs("RSN", axis=0, level=2).astype(int).to_numpy().reshape(-1)
)
uniq_rsns_lst = list(uniq_rsns)
uniq_rsns_lst.sort()

finish_times = []
istart = 0

for i in tqdm(range(istart, len(uniq_rsns_lst))):
    rsn = uniq_rsns_lst[i]
    gm_filename = retrieve_peer_gm_data(rsn)[0]
    gm_data = import_PEER(gm_filename)
    if gm_data[-1, 0] > 60.00:
        # consider trimming record
        fig, ax = plt.subplots(figsize=(12, 2.5))
        ax.plot(gm_data[:, 0], gm_data[:, 1])
        ax.grid(which="both", linewidth=0.30)
        plt.show()
        inp = input(f"[{i}/{len(uniq_rsns_lst)}]: Finish time =?")
        plt.close()
        if inp == "":
            finish_times.append(None)
        else:
            try:
                finish_times.append(float(inp))
            except (ValueError, TypeError) as e:
                print("Error with input:")
                print(e)
                break
    else:
        finish_times.append(None)

df_trim = pd.Series(finish_times, index=uniq_rsns_lst)
df_trim.to_csv("results/site_hazard/record_trim.csv")
