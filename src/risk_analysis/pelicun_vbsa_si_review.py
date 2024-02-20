"""
  Perform VBSA using pelicun
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.2f}".format


if __name__ == "__main__":

    df = pd.read_hdf("results/risk/si_db.hdf5", key="df")

    print(
        (
            df.loc[
                ("smrf", "3", "ii", "healthcare", "low", "0.4", "Cost"), "sT"
            ].unstack()
        )
        .loc[:, ("EDP", "C-DS", "B-DS", "B-DSc", "B-DSe", "B-DSi")]
        .to_string()
    )


df_plt = df
df_plt = df_plt.loc[:, "sT"].unstack()

mask = (
    (df_plt.index.get_level_values("occupancy") == "healthcare")
    & (df_plt.index.get_level_values("modeling_uncertainty") == "low")
    & (df_plt.index.get_level_values("replacement_threshold") == "0.4")
    & (df_plt.index.get_level_values("decision_variable") == "Cost")
)
df_plt = df_plt[mask]

# for thing in df_plt.index.names:
#     df_plt.loc[:, thing] = df_plt.index.get_level_values(thing)

# sns.scatterplot(data=df_plt, x='EDP', y='C-DS', hue='replacement_threshold')
# plt.plot(range(2), 'k')
# plt.show()

sns.scatterplot(data=df_plt, x="EDP", y="C-DS", hue="hazard_level", style="system")
plt.plot(range(2), "k")
plt.show()
