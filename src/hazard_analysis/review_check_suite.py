"""
Verify that the suite mean spectrum matches the target spectrum
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from osmg.ground_motion_utils import response_spectrum


# pylint: disable=invalid-name


def rotd50(ug_x, ug_y, dt=0.001, zeta=0.05):
    """
    Obtain the RotD50 spectrum of the 2D ground motion
    """
    angles = np.linspace(0.00, np.pi, 180)
    rss = np.empty((200, len(angles)))
    for i, angle in enumerate(angles):
        ug = ug_x * np.cos(angle) + ug_y * np.sin(angle)
        res = response_spectrum(ug.reshape((-1)), dt, zeta)
        rss[:, i] = res[:, 1]
    periods = res[:, 0]
    sas = np.empty(len(periods))
    for k in range(len(periods)):
        sas[k] = np.median(rss[k, :])
    # sas = np.median(rss, axis=1)
    return np.column_stack((periods, sas))


gms = [f"{i+1}" for i in range(14)]
spectra = []

for gm in tqdm(gms):
    ug_x = pd.read_csv(
        f"data/ground_motions/parsed/{gm}x.txt", header=None
    ).to_numpy()
    ug_y = pd.read_csv(
        f"data/ground_motions/parsed/{gm}y.txt", header=None
    ).to_numpy()
    results = rotd50(ug_x, ug_y, 0.005, 0.05)
    spectra.append(results)


spectra_cols = np.empty((np.shape(spectra[0][:, 1])[0], 14))
for i, spec in enumerate(spectra):
    spectra_cols[:, i] = spec[:, 1]

# load the response spectra from PEER

peer_spectra = []
for gm in gms:
    spectrum = pd.read_csv(
        f"data/ground_motions/parsed/{gm}RS.txt", header=None, sep=" "
    ).to_numpy(dtype=float)
    peer_spectra.append(spectrum)

df_rec_idx = pd.read_csv(
    "results/site_hazard/required_records.csv", index_col=0, header=[0, 1]
).index

fig, axs = plt.subplots(7, 2, sharex=True, figsize=(6, 8), sharey=True)
j = 0
for i in range(7):
    for k in range(2):
        if j == 13:
            lab1 = "flatfile"
            lab2 = "our calcs"
        else:
            lab1 = None
            lab2 = None
        ax = axs[i, k]
        ax.plot(peer_spectra[j][:, 0], peer_spectra[j][:, 1], label=lab1)
        ax.plot(spectra[j][:, 0], spectra[j][:, 1], linestyle="dashed", label=lab2)
        ax.set(xscale="log")
        ax.set(yscale="log")
        ax.grid(which="both", linewidth=0.30)
        ax.text(
            0.02,
            0.02,
            df_rec_idx[j],
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )
        j += 1
axs[-1].legend()
plt.savefig("figures/gm_response_spectra_check.pdf")
# plt.show()
plt.tight_layout()
plt.close()
