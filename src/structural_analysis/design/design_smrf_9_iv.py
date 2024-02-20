"""
Design of a 9-story SMRF risk category IV system
"""

import numpy as np
from src.structural_analysis.design.smrf_design import design_smrf_lrrs
from src.structural_analysis.design.smrf_compact_sections import family_27
from src.structural_analysis.design.smrf_compact_sections import family_33
from src.structural_analysis.design.smrf_compact_sections import family_36

beams: dict[int, list[str]] = {}
beams[1] = family_36
beams[2] = family_36
beams[3] = family_36
beams[4] = family_36
beams[5] = family_36
beams[6] = family_36
beams[7] = family_36
beams[8] = family_33
beams[9] = family_27
cols_int = family_27
cols_ext = family_27

beams2: dict[int, list[str]] = {}
beams2[1] = family_27
beams2[2] = family_27
beams2[3] = family_27
beams2[4] = family_27
beams2[5] = family_27
beams2[6] = family_27
beams2[7] = family_27
beams2[8] = family_27
beams2[9] = family_27
cols_int2 = family_27
cols_ext2 = family_27


#   lvl: 1    2    3    4    5    6    7    8    9
coeff = [
    7,
    7,
    7,
    7,
    6,
    4,
    1,
    1,
    0,  # beams
    15,
    15,
    15,
    15,
    15,
    12,
    12,
    8,
    8,  # interior
    10,
    10,
    9,
    9,
    7,
    7,
    5,
    5,
    2,  # exterior
    11,
    11,
    11,
    9,
    7,
    6,
    3,
    1,
    0,  # beams
    15,
    15,
    14,
    14,
    10,
    10,
    7,
    7,
    7,  # interior
    8,
    8,
    8,
    8,
    4,
    3,
    3,
    0,
    0,
]  # exterior


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
lvl_weight = dict(
    level_1=1146.614555 * 1e3 / 2.0,
    level_2=1129.948062 * 1e3 / 2.0,
    level_3=1131.672749 * 1e3 / 2.0,
    level_4=1135.508527 * 1e3 / 2.0,
    level_5=1131.540099 * 1e3 / 2.0,
    level_6=1092.867625 * 1e3 / 2.0,
    level_7=1080.550214 * 1e3 / 2.0,
    level_8=1067.985753 * 1e3 / 2.0,
    level_9=1153.789322 * 1e3 / 2.0,
)  # lb (only the tributary weight for this frame)

# design parameters
design_params: dict = {}
design_params["Cd"] = 5.5
design_params["R"] = 8.0
design_params["Ie"] = 1.5
design_params["ecc_ampl"] = 1.1
design_params["max_drift"] = 0.01

# site characteristics
site_characteristics: dict = {}
site_characteristics["Sds"] = 1.58
site_characteristics["Sd1"] = 1.38

tmax_params: dict = {}
tmax_params["ct"] = 0.028
tmax_params["exponent"] = 0.8

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


design_smrf_lrrs(
    num_lvls=9,
    beams=beams,
    cols_int=cols_int,
    cols_ext=cols_ext,
    beams2=beams2,
    cols_int2=cols_int2,
    cols_ext2=cols_ext2,
    coeff=coeff,
    beam_udls_dead=beam_udls_dead,
    beam_udls_live=beam_udls_live,
    lvl_weight=lvl_weight,
    design_params=design_params,
    site_characteristics=site_characteristics,
    tmax_params=tmax_params,
    mlp_periods=mlp_periods,
    mlp_des_spc=mlp_des_spc,
    risk_category="iv",
    full_results=True,
)
