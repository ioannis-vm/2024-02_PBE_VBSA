"""
Generation of conditional mean spectra using OpenSHA PSHA output.
"""

# Imports

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from src.util import read_study_param

# pylint: disable = invalid-name

spec_data_path = "results/site_hazard"

# Uniform Hazard Spectra
m = int(read_study_param("data/study_vars/m"))
uhss = [
    pd.read_csv(f"results/site_hazard/UHS_{i+1}.csv", index_col=0) for i in range(m)
]
uhss = pd.concat(uhss, axis=1)
uhss.columns = [f"{i+1}" for i in range(m)]

# adjust for directivity using the Bayless and Somerville 2013 model.
bay_coeff = np.array(
    [
        [0.5, 0.0, 0.0],
        [0.75, 0.0, 0.0],
        [1.0, -0.12, 0.075],
        [1.5, -0.175, 0.09],
        [2.0, -0.21, 0.095],
        [3.0, -0.235, 0.099],
        [4.0, -0.255, 0.103],
        [5.0, -0.275, 0.108],
        [7.5, -0.29, 0.112],
        [10.0, -0.3, 0.115],
    ]
)
fgeom = np.log(
    np.array(
        [
            37.76,
            22.27,
            16.40,
            12.80,
            10.52,
            8.90,
            7.97,
            7.23,
            6.67,
            6.22,
            5.90,
            5.57,
            5.38,
            5.18,
            4.97,
            4.86,
        ]
    )
)
fd = bay_coeff[:, 1] + bay_coeff[:, 2].reshape((-1, 1)).T * fgeom.reshape((-1, 1))
f_fd = []
for i in range(m):
    f_fd.append(
        interp1d(
            bay_coeff[:, 0],
            fd[i, :],
            kind="linear",
            fill_value=0.00,
            bounds_error=False,
        )
    )

for i, col in enumerate(uhss.columns):
    uhss[col] = uhss[col] * np.exp(f_fd[i](uhss.index.to_numpy()))

# # Store target spectra in the PEER-compatible input format
# for i in range(m):
#     np.savetxt(
#         f'{spec_data_path}/spectrum_'+str(i+1)+'.csv',
#         np.column_stack((uhss.index.to_numpy(), uhss.iloc[:, i])),
#         header='Hazard Level '+str(i+1)+',\r\n,\r\nT (s),Sa (g)',
#         delimiter=',', comments='', fmt='%.5f', newline='\r\n')

# Store target spectra
for i, uhs in enumerate(uhss):
    uhss[uhs].to_csv(f"{spec_data_path}/spectrum_{i+1}.csv")


# plot spectra

uhs_df = [
    pd.read_csv(f"results/site_hazard/UHS_{i+1}.csv", index_col=0, header=[0])
    for i in range(m)
]

archetypes = [
    f"{a}_{s}_{r}"
    for a in ("smrf", "scbf", "brbf")
    for s in ("3", "6", "9")
    for r in ("ii", "iv")
]
fig, ax = plt.subplots()
for uhs in uhs_df:
    ax.plot(uhs, "k")
for col in uhss.columns:
    ax.plot(uhss[col], "k", linestyle="dashed")
for i, archetype in enumerate(archetypes):
    if i == 0:
        label = "T_1 of archetypes"
    else:
        label = None
    base_period = float(read_study_param(f"data/{archetype}/period"))
    ax.axvline(x=base_period, color="black", linewidth=0.50, label=label)
ax.grid(which="both", linewidth=0.30)
ax.set(xscale="log", yscale="log")
ax.set(xlabel="Period [s]")
ax.set(ylabel="RotD50 Sa [g]")
ax.set(title="Uniform hazard spectra.")
plt.savefig("figures/uhss_directivity.pdf")
# plt.show()
plt.close()
