"""
Make sure that the cusom fragility and loss databases contain complete
information for the components used in all archetypes.
"""

import os
import pandas as pd

# parameters
archetype = "smrf_3_ii"
occupancy = "office"


cmp_data_path = (
    f"data/{archetype}/performance/" f"{occupancy}/input_cmp_quant_pel.csv"
)
if os.path.exists(cmp_data_path):
    # load the performance model for the specific archetype
    cmp_marginals = pd.read_csv(cmp_data_path, index_col=0)

    # load the custom fragility curve database
    damage_db = pd.read_csv(
        "data/performance/input_fragility_pelicun.csv", header=[0, 1], index_col=0
    )

    def rm_unnamed(string):
        """
        Fix column names after import
        """
        if "Unnamed: " in string:
            return ""
        return string

    damage_db.rename(columns=rm_unnamed, level=1, inplace=True)

    # load the custom loss database
    loss_db = pd.read_csv(
        "data/performance/input_loss_pelicun.csv", header=[0, 1], index_col=0
    )
    loss_db.rename(columns=rm_unnamed, level=1, inplace=True)

    cmp_list = cmp_marginals.index.unique()[:-3]
    for comp in cmp_list:
        if comp not in damage_db.index:
            print(
                archetype, occupancy, f"Component {comp} not present in damage_db."
            )
        else:
            if not damage_db.at[comp, ("Incomplete", "")] == 0:
                print(f"Component {comp} has incomplete data in damage_db")
        if comp not in loss_db.index:
            print(archetype, occupancy, f"Component {comp} not present in loss_db.")
else:
    print(archetype, occupancy, "Performance model data not found.")
