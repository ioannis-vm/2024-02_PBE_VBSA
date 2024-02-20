"""
Design of a 6-story SMRF risk category II system
"""

import numpy as np
from src.structural_analysis.design.smrf_design import design_smrf_lrrs
from src.structural_analysis.design.smrf_compact_sections import family_27
from src.structural_analysis.design.smrf_compact_sections import family_30
from src.structural_analysis.design.smrf_compact_sections import family_33

beams: dict[int, list[str]] = {}
beams[1] = family_33
beams[2] = family_33
beams[3] = family_30
beams[4] = family_30
beams[5] = family_27
beams[6] = family_27
cols_int = family_27
cols_ext = family_27


#   lvl: 1    2    3    4    5    6
coeff = [
    0,
    0,
    0,
    0,
    0,
    0,  # beams
    8,
    8,
    5,
    5,
    3,
    3,  # interior
    2,
    2,
    1,
    1,
    0,
    0,
]  # exterior


beam_udls_dead = {
    "level_1": (12.5 * (38.0 + 15.0) + 14.0 * (15.0)) / 12.0,
    "level_2": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_3": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
    "level_4": (12.5 * (38.0 + 15.0) + 13.0 * (15.0)) / 12.0,
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
lvl_weight = dict(
    level_1=1087.289497 * 1e3 / 2.0,
    level_2=1075.759778 * 1e3 / 2.0,
    level_3=1068.798496 * 1e3 / 2.0,
    level_4=1065.981046 * 1e3 / 2.0,
    level_5=1061.929727 * 1e3 / 2.0,
    level_6=1211.687073 * 1e3 / 2.0,
)  # lb (only the tributary weight for this frame)

# design parameters
design_params: dict = {}
design_params["Cd"] = 5.5
design_params["R"] = 8.0
design_params["Ie"] = 1.0
design_params["ecc_ampl"] = 1.1
design_params["max_drift"] = 0.02

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
    num_lvls=6,
    beams=beams,
    cols_int=cols_int,
    cols_ext=cols_ext,
    beams2=None,
    cols_int2=None,
    cols_ext2=None,
    coeff=coeff,
    beam_udls_dead=beam_udls_dead,
    beam_udls_live=beam_udls_live,
    lvl_weight=lvl_weight,
    design_params=design_params,
    site_characteristics=site_characteristics,
    tmax_params=tmax_params,
    mlp_periods=mlp_periods,
    mlp_des_spc=mlp_des_spc,
    risk_category="ii",
    full_results=True,
)
