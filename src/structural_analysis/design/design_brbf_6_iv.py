"""
Design of a 3-story SCBF risk category II system
"""

import numpy as np
from src.structural_analysis.design.brbf_design import design_brbf_lrrs
from src.structural_analysis.design.scbf_compact_sections import family_14
from src.structural_analysis.design.scbf_compact_sections import family_18


# Stage: Section groups and assignments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

beams: dict[int, list[str]] = {}
beams[1] = family_18
beams[2] = family_18
beams[3] = family_18
beams[4] = family_18
beams[5] = family_18
beams[6] = family_18

columns: dict[int, list[str]] = {}
columns[1] = family_14
columns[2] = family_14
columns[3] = family_14
columns[4] = family_14
columns[5] = family_14
columns[6] = family_14

brace_core_areas: dict[int, float] = {}
brace_core_areas[1] = 10.00  # in2
brace_core_areas[2] = 9.00  # in2
brace_core_areas[3] = 8.00  # in2
brace_core_areas[4] = 6.50  # in2
brace_core_areas[5] = 5.25  # in2
brace_core_areas[6] = 2.75  # in


#   lvl: 1     2     3     4     5     6
coeff = [11, 10, 9, 9, 8, 0, 7, 7, 5, 5, 3, 3]  # beams  # columns

beam_udls_dead = {
    "level_1": (12.5 * (38.0 + 15.0) + 14.0 * (15.0)) / 12.0,
    "level_2": (12.5 * (38.0 + 15.0) + 14.0 * (15.0)) / 12.0,
    "level_3": (12.5 * (38.0 + 15.0) + 14.0 * (15.0)) / 12.0,
    "level_4": (12.5 * (38.0 + 15.0) + 14.0 * (15.0)) / 12.0,
    "level_5": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_6": (12.5 * (38.0 + 15.0) + 6.5 * (15.0)) / 12.0,
}
beam_udls_live = {
    "level_1": 41.667,
    "level_2": 41.667,
    "level_3": 41.667,
    "level_4": 41.667,
    "level_5": 41.667,
    "level_6": 41.667,
}
# weight corresponds to the entire story
lvl_weight = dict(
    level_1=1100.217793 * 1e3,
    level_2=1087.387568 * 1e3,
    level_3=1094.132234 * 1e3,
    level_4=1072.253663 * 1e3,
    level_5=1068.894564 * 1e3,
    level_6=1161.128275 * 1e3,
)  # lb

design_params: dict = {}
design_params["Cd"] = 5.0
design_params["R"] = 8.0
design_params["Ie"] = 1.5
design_params["ecc_ampl"] = 1.1
design_params["max_drift"] = 0.01

# site characteristics
site_characteristics: dict = {}
site_characteristics["Sds"] = 1.58
site_characteristics["Sd1"] = 1.38

tmax_params: dict = {}
tmax_params["ct"] = 0.03
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

design_brbf_lrrs(
    num_lvls=6,
    beams=beams,
    columns=columns,
    coeff=coeff,
    brace_core_areas=brace_core_areas,
    beam_udls_dead=beam_udls_dead,
    beam_udls_live=beam_udls_live,
    lvl_weight=lvl_weight,
    design_params=design_params,
    site_characteristics=site_characteristics,
    tmax_params=tmax_params,
    mlp_periods=mlp_periods,
    mlp_des_spc=mlp_des_spc,
    num_braces=4,
)
