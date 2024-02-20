"""
Plot selected ground motion records
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


num_gms = 14

base_path = "data/ground_motions/parsed"
gm_files_x = [f"{base_path}/{i+1}x.txt" for i in range(num_gms)]
gm_files_y = [f"{base_path}/{i+1}y.txt" for i in range(num_gms)]

# hazard level to apply scaling factor
hz = 8
df_scaling = pd.read_csv(
    "results/site_hazard/required_records.csv", index_col=0, header=[0, 1]
)
scaling_factors = df_scaling[f"{hz}"]["scaling"]
rsns = list(scaling_factors.index)

fig, ax = plt.subplots(num_gms, 1, sharex=True, sharey=True, figsize=(6, 8))

for i, gm_file_x, gm_file_y, scaling in zip(
    range(num_gms), gm_files_x, gm_files_y, scaling_factors
):
    gm_data_x = pd.read_csv(gm_file_x, header=None).to_numpy() * scaling
    gm_time_x = np.linspace(0.00, 0.005 * (len(gm_data_x) - 1.00), len(gm_data_x))
    gm_data_y = pd.read_csv(gm_file_y, header=None).to_numpy() * scaling
    gm_time_y = np.linspace(0.00, 0.005 * (len(gm_data_y) - 1.00), len(gm_data_y))
    ax[i].plot(gm_time_x, gm_data_x, color="blue", alpha=0.5, linewidth=0.10)
    ax[i].plot(gm_time_y, gm_data_y, color="red", alpha=0.5, linewidth=0.10)
    ax[i].set(ylabel=rsns[i])

# plt.show()
plt.savefig("figures/ground_motion_ths.pdf")
plt.close()
