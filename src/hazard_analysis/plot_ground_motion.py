"""
Plots a ground motion file
"""

import matplotlib.pyplot as plt
from osmg.ground_motion_utils import import_PEER
from src.util import retrieve_peer_gm_data


rsn = 5657

gm_filenames = retrieve_peer_gm_data(rsn)
gm_data_x = import_PEER(gm_filenames[0])
gm_data_y = import_PEER(gm_filenames[1])

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
ax[0].plot(gm_data_x[:, 0], gm_data_x[:, 1], "k")
ax[1].plot(gm_data_y[:, 0], gm_data_y[:, 1], "k")
for axx, dr in zip(ax, ("x", "y")):
    axx.grid(which="both", linewidth=0.30)
    axx.set(ylabel=f"Acc {dr} [g]")
ax[1].set(xlabel="Time [s]")
plt.show()
