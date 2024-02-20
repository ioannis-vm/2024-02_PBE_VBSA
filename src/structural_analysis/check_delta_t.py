"""
Obtain the first mode periods of the archetypes, determine what the
maxmimum dt for time-history analysis can be.
"""

import importlib
import numpy as np
from osmg import solver


codes = ("smrf", "scbf", "brbf")
stories = ("3", "6", "9")
rcs = ("ii", "iv")
cases = [f"{c}_{s}_{r}" for c in codes for s in stories for r in rcs]

dts = []

for archetype in cases:
    archetypes_module = importlib.import_module(
        "src.structural_analysis.archetypes_2d"
    )
    try:
        archetype_builder = getattr(archetypes_module, archetype)
    except AttributeError as exc:
        raise ValueError(f"Invalid archetype code: {archetype}") from exc

    mdl, loadcase = archetype_builder("x")

    num_levels = len(mdl.levels) - 1

    modal_analysis = solver.ModalAnalysis(
        mdl, {loadcase.name: loadcase}, num_modes=num_levels * 3
    )
    modal_analysis.settings.store_forces = False
    modal_analysis.settings.store_fiber = False
    modal_analysis.settings.restrict_dof = [False, True, False, True, False]
    modal_analysis.run()

    # determine required number of modes
    print(
        archetype,
        np.cumsum(modal_analysis.modal_participation_factors(loadcase.name, "x")[1]),
    )
    nmode = np.where(
        np.cumsum(modal_analysis.modal_participation_factors(loadcase.name, "x")[1])
        > 0.95
    )[0][0]
    tn = modal_analysis.results[loadcase.name].periods[nmode]
    dtmin = 0.20 * tn
    dts.append(dtmin)

print(min(dts))
