"""
Design of a 9-story SCBF risk category II system
"""

import numpy as np
from src.structural_analysis.design.scbf_design import design_scbf_lrrs
from src.structural_analysis.design.scbf_compact_sections import family_18
from src.structural_analysis.design.scbf_compact_sections import family_14
from src.structural_analysis.design.scbf_compact_sections import family_brace


# Stage: Section groups and assignments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

beams: dict[int, list[str]] = {}
beams[1] = family_18
beams[2] = family_18
beams[3] = family_18
beams[4] = family_18
beams[5] = family_18
beams[6] = family_18
beams[7] = family_18
beams[8] = family_18
beams[9] = family_18
columns: dict[int, list[str]] = {}
columns[1] = family_14
columns[2] = family_14
columns[3] = family_14
columns[4] = family_14
columns[5] = family_14
columns[6] = family_14
columns[7] = family_14
columns[8] = family_14
columns[9] = family_14
braces: dict[int, list[str]] = {}
braces[1] = family_brace
braces[2] = family_brace
braces[3] = family_brace
braces[4] = family_brace
braces[5] = family_brace
braces[6] = family_brace
braces[7] = family_brace
braces[8] = family_brace
braces[9] = family_brace


# for i, br in enumerate(family_24):
#     print(i, br)


#   lvl: 1     2     3     4     5     6     7     8     9
coeff = [
    10,
    10,
    9,
    9,
    9,
    9,
    8,
    8,
    0,  # beams
    16,
    16,
    13,
    13,
    9,
    9,
    7,
    7,
    7,  # columns
    18,
    17,
    17,
    17,
    17,
    15,
    15,
    13,
    13,
]  # braces


beam_udls_dead = {
    "level_1": (12.5 * (38.0 + 15.0) + 14.0 * (15.0)) / 12.0,
    "level_2": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_3": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_4": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_5": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_6": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_7": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_8": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_9": (12.5 * (38.0 + 15.0) + 6.5 * (15.0)) / 12.0,
}
beam_udls_live = {
    "level_1": 41.667,
    "level_2": 41.667,
    "level_3": 41.667,
    "level_4": 41.667,
    "level_5": 41.667,
    "level_6": 41.667,
    "level_7": 41.667,
    "level_8": 41.667,
    "level_9": 41.667,
}
# weight corresponds to the entire story
lvl_weight = dict(
    level_1=1027.266878 * 1e3,
    level_2=1020.737578 * 1e3,
    level_3=1020.963116 * 1e3,
    level_4=1020.838183 * 1e3,
    level_5=1021.489558 * 1e3,
    level_6=1025.176048 * 1e3,
    level_7=1024.136830 * 1e3,
    level_8=1025.580601 * 1e3,
    level_9=1130.669864 * 1e3,
)  # lb

# design parameters
design_params: dict = {}
design_params["Cd"] = 5.0
design_params["R"] = 6.0
design_params["Ie"] = 1.0
design_params["ecc_ampl"] = 1.1
design_params["max_drift"] = 0.02

# site characteristics
site_characteristics: dict = {}
site_characteristics["Sds"] = 1.58
site_characteristics["Sd1"] = 1.38

tmax_params: dict = {}
tmax_params["ct"] = 0.02
tmax_params["exponent"] = 0.75

# multi-period design spectrum
mlp_periods = np.array(
    (
        0.00,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        5.0,
        7.5,
        10.0,
    )
)
mlp_des_spc = np.array(
    (
        0.66,
        0.66,
        0.66,
        0.67,
        0.74,
        0.90,
        1.03,
        1.22,
        1.36,
        1.48,
        1.62,
        1.75,
        1.73,
        1.51,
        1.32,
        0.98,
        0.77,
        0.51,
        0.35,
        0.26,
        0.14,
        0.083,
    )
)


design_scbf_lrrs(
    num_lvls=9,
    beams=beams,
    columns=columns,
    braces=braces,
    coeff=coeff,
    beam_udls_dead=beam_udls_dead,
    beam_udls_live=beam_udls_live,
    lvl_weight=lvl_weight,
    design_params=design_params,
    site_characteristics=site_characteristics,
    tmax_params=tmax_params,
    mlp_periods=mlp_periods,
    mlp_des_spc=mlp_des_spc,
    num_braces=2,
    show_metadata=True,
)
