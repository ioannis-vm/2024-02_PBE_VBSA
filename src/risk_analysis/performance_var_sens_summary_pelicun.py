"""
Gather and summarize variance-based sensitivity analysis results
"""

import pandas as pd
import matplotlib.pyplot as plt

# pylint: disable=invalid-name

# cases = ['healthcare3', 'office3']
cases = ["smrf_3_ii"]
occupancy = "healthcare"
# beta_m_cases = ['medium', 'low']
beta_m_cases = ["low"]
# repl_threshold_cases = [0.4, 1.0]
repl_threshold_cases = [1.00]
loss_type = "Cost"
idx = pd.IndexSlice

# hz_lvls = [f'hazard_level_{i+1}' for i in range(16)]
hz_lvls = [f"{i+1}" for i in range(8)]
rvgroups = ["EDP", "CMP", "C-DS", "C-DV", "B-DS", "B-DV"]

for case in cases:
    for beta_m_case in beta_m_cases:
        for repl in repl_threshold_cases:
            si_dfs = []  # initialize

            for hz in hz_lvls:
                for rvgroup in rvgroups:
                    res_path = (
                        f"results/risk/vbsa_{case}_"
                        f"{occupancy}_{beta_m_case}_{repl}_"
                        f"{loss_type}_{hz}_{rvgroup}.csv"
                    )
                    data = pd.read_csv(res_path, index_col=0)
                    data.index = pd.MultiIndex.from_tuples(
                        [(hz, x) for x in data.index]
                    )
                    si_dfs.append(data)

            all_df = pd.concat(si_dfs)
            all_df.index.names = ["hazard level", "RV group"]


# python plot


def bar_plot(
    ax, data, errors, data2, errors2, colors=None, total_width=0.8, single_width=1
):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the
        names of the data, the items is a list of the values.

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        values2 = data2[name]
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, (y, z) in enumerate(zip(values, values2)):
            barplot = ax.bar(
                x + x_offset,
                y,
                yerr=errors[name][x],
                width=bar_width * single_width,
                color="white",
                edgecolor="k",
                linewidth=0.3,
            )
            ax.bar(
                x + x_offset,
                z - y,
                bottom=y,
                yerr=errors2[name][x],
                width=bar_width * single_width,
                color=f"{(30.00-i)/30.00}",
                edgecolor="k",
            )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(barplot[0])

    # change the style of the axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set(ylim=(0.0, 1.0))
    # add some space between the axis and the plot
    ax.spines["left"].set_position(("outward", 8))
    ax.spines["bottom"].set_position(("outward", 5))


data = {}
erbr = {}
data2 = {}
erbr2 = {}

my_order = ["EDP", "C-DS", "C-DV", "CMP", "B-DV", "B-DS"]
my_order_names = ["EDP", "C-DM", "C-DV", "C-QNT", "B-DV", "B-DM"]
hz_lvls_names = [f"{i+1}" for i in range(8)]

for i, hzlvl in enumerate(hz_lvls):
    vals = []
    ers = []
    vals2 = []
    ers2 = []
    for rvgroup in my_order:
        vals.append(all_df.loc[(hzlvl, rvgroup), "s1"])
        ers.append(
            (
                all_df.loc[(hzlvl, rvgroup), "s1_CI_h"]
                - all_df.loc[(hzlvl, rvgroup), "s1_CI_l"]
            )
            / 2.00
        )
        vals2.append(all_df.loc[(hzlvl, rvgroup), "sT"])
        ers2.append(
            (
                all_df.loc[(hzlvl, rvgroup), "sT_CI_h"]
                - all_df.loc[(hzlvl, rvgroup), "sT_CI_l"]
            )
            / 2.00
        )
    data[hz_lvls_names[i]] = vals
    erbr[hz_lvls_names[i]] = ers
    data2[hz_lvls_names[i]] = vals2
    erbr2[hz_lvls_names[i]] = ers2

fig, ax1 = plt.subplots(1, 1, figsize=(8, 2))
bar_plot(ax1, data, erbr, data2, erbr2, total_width=0.8, single_width=1.0)
ax1.grid(which="both", axis="y", linewidth=0.30)
ax1.set_xticks(range(6), my_order_names)
plt.tight_layout()
plt.show()
