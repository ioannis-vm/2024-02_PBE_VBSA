"""
Obtain the first mode periods of the archetypes
"""

import importlib
import numpy as np
import pandas as pd
from osmg import solver
from tqdm import tqdm
from src.util import read_study_param

codes = ("smrf", "scbf", "brbf")
stories = ("3", "6", "9")
rcs = ("ii", "iv")
cases = [f"{c}_{s}_{r}" for c in codes for s in stories for r in rcs]

periods_2d = []
periods_3d = []

for archetype in tqdm(cases):
    archetypes_module = importlib.import_module(
        "src.structural_analysis.archetypes_2d"
    )
    try:
        archetype_builder = getattr(archetypes_module, archetype)
    except AttributeError as exc:
        raise ValueError(f"Invalid archetype code: {archetype}") from exc

    mdl, loadcase = archetype_builder("x")
    # from osmg.graphics.preprocessing_3d import show
    # show(mdl, loadcase)

    num_levels = len(mdl.levels) - 1

    modal_analysis = solver.ModalAnalysis(
        mdl, {loadcase.name: loadcase}, num_modes=1
    )
    modal_analysis.settings.store_forces = False
    modal_analysis.settings.store_fiber = False
    modal_analysis.settings.restrict_dof = [False, True, False, True, False]
    modal_analysis.run()

    periods_2d.append(modal_analysis.results[loadcase.name].periods[0])

for archetype in tqdm(cases):
    archetypes_module = importlib.import_module("src.structural_analysis.archetypes")
    try:
        archetype_builder = getattr(archetypes_module, archetype)
    except AttributeError as exc:
        raise ValueError(f"Invalid archetype code: {archetype}") from exc

    mdl, loadcase = archetype_builder()
    # from osmg.graphics.preprocessing_3d import show
    # show(mdl, loadcase)

    num_levels = len(mdl.levels) - 1

    modal_analysis = solver.ModalAnalysis(
        mdl, {loadcase.name: loadcase}, num_modes=1
    )
    modal_analysis.settings.store_forces = False
    modal_analysis.settings.store_fiber = False
    modal_analysis.settings.restrict_dof = [False, True, False, True, False, True]
    modal_analysis.run()

    periods_3d.append(modal_analysis.results[loadcase.name].periods[0])


design_periods = []
for archetype in cases:
    path = f"results/design_logs/{archetype}.txt"
    with open(path, "r", encoding="utf-8") as f:
        contents = f.read()
    if archetype.startswith("smrf"):
        contents = contents.split("periods: [")[1]
        contents = contents.split(" ")[0]
    else:
        contents = contents.split("T_modal = ")[1]
        contents = contents.split(" s")[0]
    design_periods.append(float(contents))


def find_closest_elements(vector, reference_vals):
    """
    Finds the closest element in reference_vals for each element in the given vector.

    Args:
        vector (ndarray): The input vector.
        reference_vals (ndarray): The reference values to compare against.

    Returns:
        ndarray: An array containing the closest element from
          reference_vals for each element in the input vector.
    """
    res = np.zeros_like(vector)

    for i, val in enumerate(vector):
        closest_element = reference_vals[np.abs(reference_vals - val).argmin()]
        res[i] = closest_element

    return res


ref_vec = np.array(
    (
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.20,
        0.25,
        0.3,
        0.4,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
    )
)

df = pd.DataFrame({"nonlinear_2d": periods_2d}, index=cases)
print(df)

df["nonlinear_3d"] = periods_3d
df["closest"] = find_closest_elements(periods_3d, ref_vec)
df["design"] = design_periods

periods_2d = []
for archetype in cases:
    periods_2d.append(read_study_param(f"data/{archetype}/period_closest"))

df["in_data"] = periods_2d

df = df.loc[:, ("design", "nonlinear_2d", "nonlinear_3d", "closest", "in_data")]

print(df)

# # update the preiods in data
# for design in ('smrf', 'scbf', 'brbf'):
#     for stor in ('3', '6', '9'):
#         for rc in ('ii', 'iv'):
#             archetype = f'{design}_{stor}_{rc}'
#             data_path_design = f'data/{archetype}/period'
#             data_path_closest = f'data/{archetype}/period_closest'
#             with open(data_path_design, 'w') as f:
#                 f.write(f'{df.loc[archetype, "design"]}')
#                 with open(data_path_closest, 'w') as f:
#                     f.write(f"{df.loc[archetype, 'closest']:.2f}")
