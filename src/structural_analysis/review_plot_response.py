"""
Plot buliding response
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


archetype = "scbf_9_ii"
hazard_level = "8"
gm_number = 21
plot_type = "ID"
direction = "x"

arch_code, stories, rc = archetype.split("_")

res_df = pd.read_parquet(
    f"results/response/{archetype}/individual_files/{hazard_level}/"
    f"gm{gm_number}/results.parquet"
)

res_df["FA"] /= 386.22

time_vec = res_df["time"]
res_df.drop(columns="time", inplace=True)

df_sub = res_df[plot_type]

num_figs = df_sub.shape[1] + 1

fig, axs = plt.subplots(num_figs, sharex=True, sharey=True)
for i, col in enumerate(df_sub):
    ax = axs[i]
    ax.plot(time_vec, df_sub[col], "k")

    # highlight peak and add value
    idx_max = np.abs(df_sub[col]).idxmax()
    ax.scatter(
        time_vec.at[idx_max],
        df_sub.at[idx_max, col],
        s=80,
        facecolor="white",
        edgecolor="black",
    )
    ax.text(
        time_vec.at[idx_max],
        df_sub.at[idx_max, col],
        f"{df_sub.at[idx_max, col]:.3f}",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.50},
        fontsize="small",
    )

    ax.grid(which="both", linewidth=0.30)
    ax.set(ylabel=f"{plot_type}-{'-'.join(col)}")

# now plot the subdivision level
dt = time_vec.at[1] - time_vec.at[0]
diff = np.round(-np.log10(np.diff(time_vec.to_numpy().reshape(-1)) / dt))
diff = np.array((0.00, *diff))
ax = axs[-1]
shay = ax.get_shared_y_axes()
shay.remove(ax)
yticker = matplotlib.axis.Ticker()
ax.yaxis.major = yticker
yloc = matplotlib.ticker.AutoLocator()
yfmt = matplotlib.ticker.ScalarFormatter()
ax.yaxis.set_major_locator(yloc)
ax.yaxis.set_major_formatter(yfmt)
ax.scatter(time_vec, diff, color="black", s=0.01)
ax.set(ylabel="subdiv. lvl")
print(f"Max subdivision level for 3D: {max(diff)}")

# second plot

if direction == "x":
    ax_incr = 0
else:
    ax_incr = 1

res_df = pd.read_parquet(
    f"results/response/{archetype}/individual_files/"
    f"{hazard_level}/gm{gm_number}/results_{direction}.parquet"
)

res_df["FA"] /= 386.22

time_vec = res_df["time"]
res_df.drop(columns="time", inplace=True)

df_sub = res_df[plot_type]

num_figs = (df_sub.shape[1]) * 2 + 1

for i, col in enumerate(df_sub):
    ax = axs[2 * i + ax_incr]
    ax.plot(time_vec, df_sub[col], "red", linestyle="dashed")

    # highlight peak and add value
    idx_max = np.abs(df_sub[col]).idxmax()
    ax.scatter(
        time_vec.at[idx_max],
        df_sub.at[idx_max, col],
        s=80,
        facecolor="white",
        edgecolor="red",
    )
    ax.text(
        time_vec.at[idx_max],
        df_sub.at[idx_max, col],
        f"{df_sub.at[idx_max, col]:.3f}",
        bbox={"facecolor": "white", "edgecolor": "red", "alpha": 0.50},
        fontsize="small",
    )

    ax.grid(which="both", linewidth=0.30)
    ax.set(ylabel=f"{plot_type}-{col[0]}")

# now plot the subdivision level
dt = time_vec.at[1] - time_vec.at[0]
diff = np.round(-np.log10(np.diff(time_vec.to_numpy().reshape(-1)) / dt))
diff = np.array((0.00, *diff))
ax = axs[-1]
shay = ax.get_shared_y_axes()
shay.remove(ax)
yticker = matplotlib.axis.Ticker()
ax.yaxis.major = yticker
yloc = matplotlib.ticker.AutoLocator()
yfmt = matplotlib.ticker.ScalarFormatter()
ax.yaxis.set_major_locator(yloc)
ax.yaxis.set_major_formatter(yfmt)
ax.scatter(time_vec, diff, color="red", s=0.01)
ax.set(ylabel="subdiv. lvl")
print(f"Max subdivision level for 2D: {max(diff)}")

# end of second plot

plt.show()
plt.close()
