"""
Compare the pushover curve of the 2d and 3d archetypes
"""

# -------------------------------------------------------
# imports

import os
import importlib
import numpy as np
import pandas as pd
from dask.distributed import Client
import dask.bag as db
from osmg import solver
from osmg.gen.query import ElmQuery

# from osmg.graphics.preprocessing_3d import show


def process_item(item):
    archetype = item[0]
    direction = item[1]

    # -------------------------------------------------------
    # parameter calculation

    system, stories, risk_category = archetype.split("_")

    if risk_category == "ii":
        peak_drift = pd.Series(
            np.array(
                (
                    16.00,
                    40.0,
                    40.0,
                    16.00,
                    30.0,
                    60.0,
                    30.00,
                    40.0,
                    60.0,
                )
            ),
            index=pd.MultiIndex.from_tuples(
                (
                    ("smrf", "3"),
                    ("smrf", "6"),
                    ("smrf", "9"),
                    ("scbf", "3"),
                    ("scbf", "6"),
                    ("scbf", "9"),
                    ("brbf", "3"),
                    ("brbf", "6"),
                    ("brbf", "9"),
                )
            ),
        )
    else:
        peak_drift = pd.Series(
            np.array(
                (
                    16.00,
                    40.0,
                    40.0,
                    20.00,
                    30.0,
                    40.0,
                    30.00,
                    40.0,
                    60.0,
                )
            ),
            index=pd.MultiIndex.from_tuples(
                (
                    ("smrf", "3"),
                    ("smrf", "6"),
                    ("smrf", "9"),
                    ("scbf", "3"),
                    ("scbf", "6"),
                    ("scbf", "9"),
                    ("brbf", "3"),
                    ("brbf", "6"),
                    ("brbf", "9"),
                )
            ),
        )

    # -------------------------------------------------------
    # load archetype building
    archetypes_module = importlib.import_module(
        "src.structural_analysis.archetypes_2d"
    )
    try:
        archetype_builder = getattr(archetypes_module, archetype)
    except AttributeError as exc:
        raise ValueError(f"Invalid archetype code: {archetype}") from exc

    mdl, loadcase = archetype_builder(direction)
    num_levels = len(mdl.levels) - 1
    level_heights = []
    for level in mdl.levels.values():
        level_heights.append(level.elevation)
    level_heights = np.diff(level_heights)

    lvl_nodes = []
    base_node = list(mdl.levels[0].nodes.values())[0]
    lvl_nodes.append(base_node)
    for i in range(num_levels):
        lvl_nodes.append(loadcase.parent_nodes[i + 1])

    # -------------------------------------------------------
    # modal analysis (to get the mode shape)

    # fix leaning column

    elmq = ElmQuery(mdl)
    for i in range(num_levels):
        nd = elmq.search_node_lvl(0.00, 0.00, i + 1)
        nd.restraint = [False, False, False, True, True, True]

    modal_analysis = solver.ModalAnalysis(
        mdl, {loadcase.name: loadcase}, num_modes=1
    )
    modal_analysis.settings.store_forces = False
    modal_analysis.settings.store_fiber = False
    modal_analysis.settings.restrict_dof = [False, True, False, True, False, True]
    modal_analysis.run()
    modeshape_lst = []
    for nd in lvl_nodes:
        modeshape_lst.append(
            modal_analysis.results[loadcase.name].node_displacements[nd.uid][0][0]
        )
    modeshape = np.array(modeshape_lst)

    # -------------------------------------------------------
    # pushover analysis

    for i in range(num_levels):
        nd = elmq.search_node_lvl(0.00, 0.00, i + 1)
        nd.restraint = [False, False, False, False, False, False]

    # define analysis
    anl = solver.PushoverAnalysis(mdl, {loadcase.name: loadcase})
    anl.settings.store_forces = False
    anl.settings.store_release_force_defo = False
    snodes = [n.uid for n in lvl_nodes + list(mdl.levels[0].nodes.values())]
    snodes.extend([x.uid for x in loadcase.parent_nodes.values()])
    anl.settings.specific_nodes = snodes
    anl.settings.solver = "UmfPack"
    anl.settings.restrict_dof = [False, True, False, True, False, True]
    control_node = lvl_nodes[-1]

    anl.run(
        "x", [peak_drift[system, stories]], control_node, 1.0, modeshape=modeshape
    )
    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # show_deformed_shape(
    #     anl, loadcase.name, anl.results[loadcase.name].n_steps_success-1,
    #     0.0, extrude=True, animation=False)

    res_df = pd.DataFrame()
    for i_story, node in enumerate(lvl_nodes):
        if i_story == 0:
            continue
        results = np.column_stack(anl.table_pushover_curve(loadcase.name, "x", node))
        if i_story == 1:
            res_df["Vb"] = results[:, 1]
        res_df[f"Level {i_story}"] = results[:, 0]
    res_df.index.name = "Step"

    # -------------------------------------------------------
    # save pushover results
    if not os.path.exists(f"results/response/{archetype}/pushover"):
        os.makedirs(f"results/response/{archetype}/pushover")
    res_df.to_csv(f"results/response/{archetype}/pushover/results_{direction}.csv")


if __name__ == "__main__":
    # Create a Dask client with a specific number of workers/cores
    client = Client(n_workers=12)  # Set the desired number of workers/cores here

    # Create a list of items to process
    items = []
    for cd in ("smrf", "scbf", "brbf"):
        for st in ("3", "6", "9"):
            for rc in ("ii", "iv"):
                for dr in ("x",):
                    items.append([f"{cd}_{st}_{rc}", dr])

    # Convert the list to a Dask bag
    bag = db.from_sequence(items)

    # Apply the function in parallel using Dask's map operation
    results = bag.map(process_item)

    # Compute the results and retrieve the output
    output = results.compute()
