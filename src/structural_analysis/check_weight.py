"""
Obtain the first mode periods of the archetypes
"""

import importlib
import pandas as pd


codes = ("smrf", "scbf", "brbf")
stories = ("3", "6", "9")
rcs = ("ii", "iv")
cases = [f"{c}_{s}_{r}" for c in codes for s in stories for r in rcs]


for archetype in cases:
    trib_fact = {
        "smrf_3_ii": 2.00,
        "smrf_6_ii": 2.00,
        "smrf_9_ii": 2.00,
        "smrf_3_iv": 2.00,
        "smrf_6_iv": 2.00,
        "smrf_9_iv": 2.00,
        "scbf_3_ii": 2.00,
        "scbf_6_ii": 2.00,
        "scbf_9_ii": 2.00,
        "scbf_3_iv": 4.00,
        "scbf_6_iv": 4.00,
        "scbf_9_iv": 4.00,
        "brbf_3_ii": 2.00,
        "brbf_6_ii": 2.00,
        "brbf_9_ii": 2.00,
        "brbf_3_iv": 4.00,
        "brbf_6_iv": 4.00,
        "brbf_9_iv": 4.00,
    }

    m2d = []
    m3d = []

    archetypes_module = importlib.import_module(
        "src.structural_analysis.archetypes_2d"
    )
    try:
        archetype_builder = getattr(archetypes_module, archetype)
    except AttributeError as exc:
        raise ValueError(f"Invalid archetype code: {archetype}") from exc

    mdl, loadcase = archetype_builder("x")

    num_lvl = len(mdl.levels) - 1
    for lvl_idx in range(num_lvl):
        level = mdl.levels[lvl_idx + 1]
        nodes = [n.uid for n in level.nodes.values()]
        for comp in level.components.values():
            nodes.extend([x.uid for x in comp.internal_nodes.values()])
        mass = sum(loadcase.node_mass[x].val[0] for x in nodes)
        mass += loadcase.node_mass[loadcase.parent_nodes[lvl_idx + 1].uid].val[0]
        m2d.append(mass)

    archetypes_module = importlib.import_module("src.structural_analysis.archetypes")
    try:
        archetype_builder = getattr(archetypes_module, archetype)
    except AttributeError as exc:
        raise ValueError(f"Invalid archetype code: {archetype}") from exc

    mdl, loadcase = archetype_builder()

    num_lvl = len(mdl.levels) - 1
    for lvl_idx in range(num_lvl):
        level = mdl.levels[lvl_idx + 1]
        nodes = [n.uid for n in level.nodes.values()]
        for comp in level.components.values():
            nodes.extend([x.uid for x in comp.internal_nodes.values()])
        mass = sum(loadcase.node_mass[x].val[0] for x in nodes)
        mass += loadcase.node_mass[loadcase.parent_nodes[lvl_idx + 1].uid].val[0]
        m3d.append(mass)

    df = pd.DataFrame({"2d": m2d, "3d": m3d}, index=range(1, num_lvl + 1))
    df["2d"] *= trib_fact[archetype]

    df["diff"] = ((df["3d"] - df["2d"])) * 386.22 / 1000.00
    print(archetype)
    print(df)
    print()
