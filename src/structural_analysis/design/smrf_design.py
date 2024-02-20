"""
Design a n-story SMRF system
"""

from typing import Optional
from osmg.model import Model
from osmg.gen.component_gen import BeamColumnGenerator
from osmg.gen.section_gen import SectionGenerator
from osmg.gen.query import ElmQuery
from osmg import defaults
from osmg.ops.element import ElasticBeamColumn
from osmg.gen.zerolength_gen import release_6
from osmg.load_case import LoadCase
from osmg import solver
from osmg.ops.section import ElasticSection
from osmg import common
from osmg.postprocessing.design import LoadCombination
from osmg.postprocessing.steel_design_checks import smrf_scwb
from osmg.postprocessing.steel_design_checks import smrf_pz_doubler_plate_requirement
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d  # type: ignore
import pandas as pd  # type: ignore

# from osmg.graphics.preprocessing_3d import show
# from osmg.graphics.postprocessing_3d import show_deformed_shape
# from osmg.graphics.postprocessing_3d import show_basic_forces
# from osmg.graphics.postprocessing_3d import show_basic_forces_combo

# pylint: disable=invalid-name
# pylint: disable=no-else-return

nparr = npt.NDArray[np.float64]


def design_smrf_lrrs(
    num_lvls,
    beams: dict[int, list[str]],
    cols_int: list[str],
    cols_ext: list[str],
    beams2: Optional[dict[int, list[str]]],
    cols_int2: Optional[list[str]],
    cols_ext2: Optional[list[str]],
    coeff: list[int],
    beam_udls_dead: dict[str, float],
    beam_udls_live: dict[str, float],
    lvl_weight: dict[str, float],
    design_params: dict[str, float],
    site_characteristics: dict[str, float],
    tmax_params: dict[str, float],
    mlp_periods: nparr,
    mlp_des_spc: nparr,
    risk_category: str,
    full_results=False,
):
    """
    Design a n-story SMRF system
    """
    # output flags
    get_doubler_plates = full_results
    get_beam_checks = full_results

    # selecting sections
    beam_coeff_lvl: dict[int, int] = {}
    col_int_coeff_lvl: dict[int, int] = {}
    col_ext_coeff_lvl: dict[int, int] = {}
    running_idx = 0
    for lvl_idx in range(1, num_lvls + 1):
        beam_coeff_lvl[lvl_idx] = running_idx
        running_idx += 1
    for lvl_idx in range(1, num_lvls + 1):
        col_int_coeff_lvl[lvl_idx] = running_idx
        running_idx += 1
    for lvl_idx in range(1, num_lvls + 1):
        col_ext_coeff_lvl[lvl_idx] = running_idx
        running_idx += 1
    if risk_category == "iv":
        beam2_coeff_lvl: Optional[dict[int, int]] = {}
        col_int2_coeff_lvl: Optional[dict[int, int]] = {}
        col_ext2_coeff_lvl: Optional[dict[int, int]] = {}
        for lvl_idx in range(1, num_lvls + 1):
            beam2_coeff_lvl[lvl_idx] = running_idx
            running_idx += 1
        for lvl_idx in range(1, num_lvls + 1):
            col_int2_coeff_lvl[lvl_idx] = running_idx
            running_idx += 1
        for lvl_idx in range(1, num_lvls + 1):
            col_ext2_coeff_lvl[lvl_idx] = running_idx
            running_idx += 1
    else:
        beam2_coeff_lvl = None
        col_int2_coeff_lvl = None
        col_ext2_coeff_lvl = None

    # initializing model

    mdl = Model(f"office_{num_lvls}_design")
    mdl.settings.imperial_units = True
    mcg = BeamColumnGenerator(mdl)
    secg = SectionGenerator(mdl)
    query = ElmQuery(mdl)

    rigidsec = secg.generate_generic_elastic("rigidsec", 1.0e12, 1.0e12, 1.0e12)

    heights = [15.00]
    heights_diff = [15.00]
    for lvl_idx in range(1, num_lvls):
        heights.append(15.00 + 13.00 * lvl_idx)
        heights_diff.append(13.00)
    hi = np.array(heights) * 12.00  # in
    hi_diff = np.array(heights_diff) * 12.00

    mdl.add_level(0, 0.00)
    for i, h in enumerate(hi):
        mdl.add_level(i + 1, h)

    defaults.load_default_steel(mdl)
    defaults.load_default_fix_release(mdl)

    def section_from_idx(lst_idx, list_of_section_names):
        """
        Define a section from a given list and index
        """
        secg.load_aisc_from_database(
            "W",
            [list_of_section_names[lst_idx]],
            "default steel",
            "default steel",
            ElasticSection,
        )
        res_sec = mdl.elastic_sections.retrieve_by_attr(
            "name", list_of_section_names[lst_idx]
        )
        return res_sec

    beam_secs: dict = {}
    col_int_secs: dict = {}
    col_ext_secs: dict = {}

    if risk_category == "iv":
        beam2_secs: Optional[dict] = {}
        col_int2_secs: Optional[dict] = {}
        col_ext2_secs: Optional[dict] = {}
    else:
        beam2_secs = None
        col_int2_secs = None
        col_ext2_secs = None

    for lvl_idx in range(1, num_lvls + 1):
        beam_secs[f"level_{lvl_idx}"] = section_from_idx(
            coeff[beam_coeff_lvl[lvl_idx]], beams[lvl_idx]
        )
        col_int_secs[f"level_{lvl_idx}"] = section_from_idx(
            coeff[col_int_coeff_lvl[lvl_idx]], cols_int
        )
        col_ext_secs[f"level_{lvl_idx}"] = section_from_idx(
            coeff[col_ext_coeff_lvl[lvl_idx]], cols_ext
        )
        if risk_category == "iv":
            assert beams2 is not None
            assert cols_int2 is not None
            assert cols_ext2 is not None
            assert beam2_secs is not None
            assert col_int2_secs is not None
            assert col_ext2_secs is not None
            beam2_secs[f"level_{lvl_idx}"] = section_from_idx(
                coeff[beam2_coeff_lvl[lvl_idx]], beams2[lvl_idx]
            )
            col_int2_secs[f"level_{lvl_idx}"] = section_from_idx(
                coeff[col_int2_coeff_lvl[lvl_idx]], cols_int2
            )
            col_ext2_secs[f"level_{lvl_idx}"] = section_from_idx(
                coeff[col_ext2_coeff_lvl[lvl_idx]], cols_ext2
            )

    print("Sections")
    print("Beams")
    print([sec.name for sec in beam_secs.values()])
    print("Interior Columns")
    print([sec.name for sec in col_int_secs.values()])
    print("Exterior Columns")
    print([sec.name for sec in col_ext_secs.values()])
    if risk_category == "iv":
        print("Sections - inner frame")
        print("Beams")
        print([sec.name for sec in beam2_secs.values()])
        print("Interior Columns")
        print([sec.name for sec in col_int2_secs.values()])
        print("Exterior Columns")
        print([sec.name for sec in col_ext2_secs.values()])
    print()

    # define structural elements
    x_locs = np.array([0.00, 25.00, 50.00, 75.00, 100.00]) * 12.00  # (in)
    x_locs_2 = np.array([150.00, 175.00, 200.00]) * 12.00  # second frame

    for level_counter in range(num_lvls):
        # frame columns
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])
        if risk_category == "iv":
            loc_lists = (x_locs, x_locs_2)
        else:
            loc_lists = (x_locs,)
        for frno, loc_list in enumerate(loc_lists):
            for xpt in loc_list:
                if frno == 0:  # outer frame
                    if xpt in [loc_list[0], loc_list[-1]]:
                        sec = col_ext_secs[level_tag]
                    else:
                        sec = col_int_secs[level_tag]
                else:  # inner frame
                    if xpt in [loc_list[0], loc_list[-1]]:
                        sec = col_ext2_secs[level_tag]
                    else:
                        sec = col_int2_secs[level_tag]
                pt = np.array((xpt, 0.00))
                mcg.add_vertical_active(
                    pt[0],
                    pt[1],
                    np.zeros(3),
                    np.zeros(3),
                    "Linear",
                    1,
                    sec,
                    ElasticBeamColumn,
                    "centroid",
                    np.pi / 2.00,
                )

            # frame beams
            for ipt_idx in range(len(loc_list) - 1):
                pt_i = np.array((loc_list[ipt_idx], 0.00))
                pt_j = np.array((loc_list[ipt_idx + 1], 0.00))

                if frno == 0:
                    sec = beam_secs[level_tag]
                else:
                    sec = beam2_secs[level_tag]
                mcg.add_horizontal_active(
                    pt_i[0],
                    pt_i[1],
                    pt_j[0],
                    pt_j[1],
                    np.array((0.0, 0.0, 0.0)),
                    np.array((0.0, 0.0, 0.0)),
                    "centroid",
                    "centroid",
                    "Linear",
                    1,
                    sec,
                    ElasticBeamColumn,
                    "centroid",
                )

    # leaning column
    for level_counter in range(num_lvls):
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])
        pt = np.array((125.00 * 12.00, 0.00))
        mcg.add_vertical_active(
            pt[0],
            pt[1],
            np.zeros(3),
            np.zeros(3),
            "Corotational",
            1,
            rigidsec,
            ElasticBeamColumn,
            "centroid",
            np.pi / 2.00,
            method="generate_hinged_component_assembly",
            additional_args={
                "n_x": None,
                "n_y": None,
                "zerolength_gen_i": None,
                "zerolength_gen_args_i": {},
                "zerolength_gen_j": release_6,
                "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
            },
        )
        if risk_category == "iv":
            other_locs = (x_locs[-1], x_locs_2[0])
        else:
            other_locs = (x_locs[-1],)
        for other_loc in other_locs:
            pt_i = np.array((other_loc, 0.00))
            pt_j = pt
            mcg.add_horizontal_active(
                pt_i[0],
                pt_i[1],
                pt_j[0],
                pt_j[1],
                np.array((0.0, 0.0, 0.0)),
                np.array((0.0, 0.0, 0.0)),
                "centroid",
                "centroid",
                "Linear",
                1,
                rigidsec,
                ElasticBeamColumn,
                "centroid",
                method="generate_hinged_component_assembly",
                additional_args={
                    "n_x": None,
                    "n_y": None,
                    "zerolength_gen_i": release_6,
                    "zerolength_gen_args_i": {"distance": 1.00, "n_sub": 1},
                    "zerolength_gen_j": release_6,
                    "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
                },
            )

    p_nodes: list = []
    for lvl_idx in range(1, num_lvls + 1):
        p_nodes.append(query.search_node_lvl(125.00 * 12.00, 0.00, lvl_idx))

    # restrict motion in XZ plane
    for node in mdl.list_of_all_nodes():
        node.restraint = [False, True, False, True, False, True]
    # fix base
    for node in mdl.levels[0].nodes.values():
        # pin:
        # node.restraint = [True, True, True, True, False, True]
        node.restraint = [True, True, True, True, True, True]
    # # fix leaning column
    # query.search_node_lvl(125.00*12.00, 0.00, 0).restraint = [True]*6

    # ~~~~~~~~~~~~ #
    # assign loads #
    # ~~~~~~~~~~~~ #

    lc_dead = LoadCase("dead", mdl)
    self_mass(mdl, lc_dead)
    self_weight(mdl, lc_dead)
    lc_live = LoadCase("live", mdl)

    for level_counter in range(1, num_lvls + 1):
        level_tag = "level_" + str(level_counter)
        if risk_category == "iv":
            xpts = np.concatenate((x_locs[:-1], x_locs_2[:-1]))
        else:
            xpts = x_locs[:-1]
        for xpt in xpts:
            xpt += 30.00
            comp = query.retrieve_component(xpt, 0.00, level_counter)
            assert comp is not None
            for elm in comp.elements.values():
                lc_dead.line_element_udl[elm.uid].add_glob(
                    np.array((0.00, 0.00, -beam_udls_dead[level_tag]))
                )
                lc_live.line_element_udl[elm.uid].add_glob(
                    np.array((0.00, 0.00, -beam_udls_live[level_tag]))
                )
        nd = query.search_node_lvl(1200.00, 0.00, level_counter)
        assert nd is not None
        lc_dead.node_loads[nd.uid].val += np.array(
            (0.00, 0.00, -lvl_weight[level_tag], 0.00, 0.00, 0.00)
        )
        mass = lvl_weight[level_tag] / common.G_CONST_IMPERIAL
        lc_dead.node_mass[nd.uid].val += np.array(
            (mass, mass, mass, 0.00, 0.00, 0.00)
        )

    # ~~~~~~~~~~~~ #
    # run analyses #
    # ~~~~~~~~~~~~ #

    # dead and live static analysis

    static_anl = solver.StaticAnalysis(mdl, {"dead": lc_dead, "live": lc_live})
    static_anl.settings.system_overwrite = "SparseSYM"
    static_anl.run()

    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # from osmg.graphics.postprocessing_3d import show_basic_forces
    # show_deformed_shape(static_anl, 'dead', 0, 0.00, False)
    # show_basic_forces(static_anl, 'dead', 0, 1.00, 0.00, 0.00, 1.00e-1,
    #                   0.00, 10, global_axes=True)
    # earthquake - ELF (ASCE 7-22 Sec. 12.8)

    # design parameters
    Cd = design_params["Cd"]
    R = design_params["R"]
    Ie = design_params["Ie"]
    ecc_ampl = design_params["ecc_ampl"]
    max_drift = design_params["max_drift"]

    # site characteristics
    Sds = site_characteristics["Sds"]
    Sd1 = site_characteristics["Sd1"]

    def k(T):
        """
        ASCE 7-22 Sec. 12.8.3
        """
        if T <= 0.5:
            res = 1.0
        elif T >= 2.5:
            res = 2.0
        else:
            x = np.array([0.5, 2.5])
            y = np.array([1.0, 2.0])
            f = interp1d(x, y)
            res = f(np.array([T]))[0]
        return res

    def Tmax(c_t, expnt, height, s_d1):
        """
        ASCE 7-22 Sec. 12.8.2.1
        """

        def cu(s_d1):
            """
            ASCE 7-22 Table 12.8-1
            """
            if s_d1 <= 0.1:
                cu = 1.7
            elif s_d1 >= 0.4:
                cu = 1.4
            else:
                x = np.array([0.1, 0.15, 0.2, 0.3, 0.4])
                y = np.array([1.7, 1.6, 1.5, 1.4, 1.4])
                f = interp1d(x, y)
                cu = f(np.array([s_d1]))[0]
            return cu

        Ta = c_t * height**expnt
        return cu(s_d1) * Ta

    def cs(t, s_ds, s_d1, r_fact, i_e):
        """
        ASCE 7-22 Sec. 12.8.1.1
        """
        t_short = s_d1 / s_ds
        if t < t_short:
            res = s_ds / r_fact * i_e
        else:
            res = s_d1 / r_fact * i_e / t
        return res

    # period estimation (Table 12.8-2)
    ct = tmax_params["ct"]
    exponent = tmax_params["exponent"]
    T_max = Tmax(ct, exponent, hi[-1] / 12.00, Sd1)

    # print(f'T_max = {T_max:.2f} s')

    # modal period
    lc_modal = LoadCase("modal", mdl)
    lc_modal.node_mass = lc_dead.node_mass
    num_modes = num_lvls

    modal_analysis = solver.ModalAnalysis(
        mdl, {"modal": lc_modal}, num_modes=num_modes
    )
    modal_analysis.run()
    ts = modal_analysis.results["modal"].periods
    assert ts is not None
    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # show_deformed_shape(
    #     modal_analysis, 'modal', 0, 0.00,
    #     extrude=False, animation=False)
    # show_deformed_shape(
    #     modal_analysis, 'modal', 1, 0.00,
    #     extrude=False, animation=False)
    # show_deformed_shape(
    #     modal_analysis, 'modal', 2, 0.00,
    #     extrude=False, animation=False)

    # mode shape
    disps = np.zeros(len(p_nodes))
    for i, p_node in enumerate(p_nodes):
        assert p_node is not None
        disps[i] = modal_analysis.results["modal"].node_displacements[p_node.uid][0][
            0
        ]
    disps /= disps[-1]
    # print(f'T_modal = {ts[0]:.2f} s\n')
    # print()

    print("Frist mode shape")
    print(disps)
    print()

    t_use = min(ts[0], T_max)
    wi = np.array(list(lvl_weight.values()))
    print(f"Seismic weight = {np.sum(wi)/1000.00:.3f} kips")
    vb_elf = np.sum(wi) * cs(t_use, Sds, Sd1, R, Ie) * ecc_ampl
    # print(f'Seismic weight: {np.sum(wi):.0f} kips')
    # print()
    # print(f'cs = {cs(t_use, Sds, Sd1, R, Ie):.3f}')
    print(f"V_b_elf = {vb_elf/1000:.2f} kips \n")
    # print(f'Cs = {cs(t_use, Sds, Sd1, R, Ie)}')
    cvx = wi * hi ** k(ts[1]) / np.sum(wi * hi ** k(ts[1]))
    fx = vb_elf * cvx

    lc_elf = LoadCase("elf", mdl)
    for i, nd in enumerate(p_nodes):
        assert nd is not None
        lc_elf.node_loads[nd.uid].add(
            np.array((fx[i], 0.00, 0.00, 0.00, 0.00, 0.00))
        )

    elf_anl = solver.StaticAnalysis(mdl, {"elf": lc_elf})
    elf_anl.run()
    # show_deformed_shape(
    #     elf_anl, 'elf', 0, 0.00,
    #     extrude=False, animation=False)
    # show_basic_forces(
    #     elf_anl, 'elf', 0, 1.00, 0.00, 0.00, 0.00, 0.00, 5, 1.00, 1.00
    # )

    elf_combo = LoadCombination(
        mdl, {"+E": [(1.00, elf_anl, "elf")], "-E": [(-1.00, elf_anl, "elf")]}
    )
    # show_basic_forces_combo(
    #     elf_combo, 1.00, .00, .0, .0, .0, 50, global_axes=True,
    #     # force_conversion=1.00/1000.00,
    #     # moment_conversion=1.00/12.00/1000.00
    # )

    # Global stability (P-Delta Effects). ASCE 7-22 12.8.7
    # units used here: lb, in
    thetas = np.zeros(len(p_nodes))
    theta_lim = 0.10
    lvlw = np.array(list(lvl_weight.values()))
    for lvl_idx in range(len(p_nodes)):
        if lvl_idx == 0:
            deltax = np.max(
                np.abs(
                    [
                        r[0]
                        for r in elf_combo.envelope_node_displacement(
                            p_nodes[lvl_idx]
                        )
                    ]
                )
            )
        else:
            deltax = np.max(
                np.abs(
                    [
                        r[0]
                        for r in elf_combo.envelope_node_displacement_diff(
                            p_nodes[lvl_idx], p_nodes[lvl_idx - 1]
                        )
                    ]
                )
            )
        px = np.sum(lvlw[lvl_idx:])
        vx = np.sum(fx[lvl_idx:])
        hsx = hi[lvl_idx]
        thetas[lvl_idx] = (px / hsx) / (vx / deltax)
    print("P-Delta Amplification Capacity Ratios (needs to be < 1.00)")
    print(thetas / theta_lim)  # should be < 1
    print()

    rsa = solver.ModalResponseSpectrumAnalysis(
        mdl, lc_modal, num_modes, mlp_periods, mlp_des_spc, "x"
    )
    rsa.run()
    assert rsa.anl is not None
    ts = rsa.anl.results["modal"].periods
    vb_modal = np.sqrt(np.sum(rsa.vb_modal**2)) / 1000 / (R / Ie) * ecc_ampl
    print(f"V_b_modal = {vb_modal:.2f} kips \n")

    vb_ampl = (vb_elf / 1000.00) / vb_modal
    drift_combo = LoadCombination(
        mdl,
        {
            "D+L": [(1.00, static_anl, "dead"), (0.50 * 0.4, static_anl, "live")],
            "+E": [(1.00 / (R / Ie) * ecc_ampl, rsa, "modal")],  # type: ignore
            "-E": [(-1.00 / (R / Ie) * ecc_ampl, rsa, "modal")],  # type: ignore
        },
    )  # coeffs from ASCE 7-22 12.8.6.1

    # show_basic_forces_combo(
    #     drift_combo, 1.00, .00, .0, .0, .0, 50, global_axes=True,
    #     # force_conversion=1.00/1000.00,
    #     # moment_conversion=1.00/12.00/1000.00
    # )
    dr: dict = {}

    dr[1] = (
        np.max(
            np.abs(
                [r[0] for r in drift_combo.envelope_node_displacement(p_nodes[0])]
            )
        )
        / (15.0 * 12.0)
        * Cd
        / Ie
    )
    for lvl_idx in range(2, num_lvls + 1):
        dr[lvl_idx] = (
            np.max(
                np.abs(
                    [
                        r[0]
                        for r in drift_combo.envelope_node_displacement_diff(
                            p_nodes[lvl_idx - 1], p_nodes[lvl_idx - 2]
                        )
                    ]
                )
            )
            / (13.0 * 12.0)
            * Cd
            / Ie
        )

    strength_combo = LoadCombination(
        mdl,
        {
            "D+L+Ev+Eh": [
                (1.20, static_anl, "dead"),
                (0.50 * 0.4, static_anl, "live"),
                (0.20 * Sds, static_anl, "dead"),
                (1.00 / (R / Ie) * vb_ampl * ecc_ampl, rsa, "modal"),
            ],
            "D+L+Ev-Eh": [
                (1.20, static_anl, "dead"),
                (0.50 * 0.4, static_anl, "live"),
                (0.20 * Sds, static_anl, "dead"),
                (-1.00 / (R / Ie) * vb_ampl * ecc_ampl, rsa, "modal"),
            ],
            "D+L-Ev-Eh": [
                (1.20, static_anl, "dead"),
                (0.50 * 0.4, static_anl, "live"),
                (-0.20 * Sds, static_anl, "dead"),
                (-1.00 / (R / Ie) * vb_ampl * ecc_ampl, rsa, "modal"),
            ],
            "D+L-Ev+Eh": [
                (1.20, static_anl, "dead"),
                (0.50 * 0.4, static_anl, "live"),
                (-0.20 * Sds, static_anl, "dead"),
                (1.00 / (R / Ie) * vb_ampl * ecc_ampl, rsa, "modal"),
            ],
        },
    )

    # from osmg.graphics.postprocessing_3d import show_basic_forces_combo
    # show_basic_forces_combo(
    #     strength_combo, 1.00, .00, .0, .0, .0, 50, global_axes=True,
    #     force_conversion=1.00/1000.00,
    #     moment_conversion=1.00/12.00/1000.00
    # )  # kips, kip-ft

    # strong column-weak beam check

    level_tags = [f"level_{i+1}" for i in range(num_lvls)]

    col_puc: dict[str, dict[str, float]] = {"exterior": {}, "interior": {}}
    for i, level in enumerate(level_tags):
        comp = query.retrieve_component(x_locs[0], 0.00, i + 1)
        assert comp is not None
        elm = list(comp.elements.values())[0]
        axial = np.abs(drift_combo.envelope_basic_forces(elm, 2)[0]["nx"].min())
        col_puc["exterior"][level] = axial
        comp = query.retrieve_component(x_locs[1], 0.00, i + 1)
        assert comp is not None
        elm = list(comp.elements.values())[0]
        axial = np.abs(drift_combo.envelope_basic_forces(elm, 2)[0]["nx"].min())
        col_puc["interior"][level] = axial
    if risk_category == "iv":
        col_puc_int: dict[str, dict[str, float]] = {"exterior": {}, "interior": {}}
        for i, level in enumerate(level_tags):
            comp = query.retrieve_component(x_locs_2[0], 0.00, i + 1)
            assert comp is not None
            elm = list(comp.elements.values())[0]
            axial = np.abs(drift_combo.envelope_basic_forces(elm, 2)[0]["nx"].min())
            col_puc_int["exterior"][level] = axial
            comp = query.retrieve_component(x_locs_2[1], 0.00, i + 1)
            assert comp is not None
            elm = list(comp.elements.values())[0]
            axial = np.abs(drift_combo.envelope_basic_forces(elm, 2)[0]["nx"].min())
            col_puc_int["interior"][level] = axial

    # strong column - weak beam
    ext_res = []
    int_res = []

    for place in ["exterior", "interior"]:
        for level_num in range(len(level_tags) - 1):
            this_level = level_tags[level_num]
            level_above = level_tags[level_num + 1]
            beam_sec = beam_secs[this_level].properties
            # dist_a = beam_sec['bf'] * 0.625
            # dist_b = beam_sec['d'] * 0.75
            # sh = dist_a + dist_b/2.00
            sh = 0.00  # WUF-W connection
            if place == "exterior":
                col_sec = col_ext_secs[this_level].properties
                col_sec_above = col_ext_secs[level_above].properties
            else:
                col_sec = col_int_secs[this_level].properties
                col_sec_above = col_int_secs[level_above].properties
            if place == "exterior":
                capacity = smrf_scwb(
                    col_sec,
                    col_sec_above,
                    beam_sec,
                    col_puc[place][this_level],
                    (
                        1.2 * beam_udls_dead[this_level]
                        + 0.50 * 0.40 * beam_udls_live[this_level]
                    ),
                    1.00,
                    hi_diff[level_num],
                    25.00 * 12.00,
                    None,
                    None,
                    None,
                    sh,
                    50000.00,
                )
                ext_res.append(capacity)
            if place == "interior":
                capacity = smrf_scwb(
                    col_sec,
                    col_sec_above,
                    beam_sec,
                    col_puc[place][this_level],
                    (
                        beam_udls_dead[this_level]
                        + 0.50 * 0.40 * beam_udls_live[this_level]
                    ),
                    1.00,
                    hi_diff[level_num],
                    25.00 * 12.00,
                    beam_sec,
                    (
                        1.2 * beam_udls_dead[this_level]
                        + 0.50 * 0.40 * beam_udls_live[this_level]
                    ),
                    1.00,
                    sh,
                    50000.00,
                )
                int_res.append(capacity)
    scwb_check = pd.DataFrame(
        {"exterior": ext_res, "interior": int_res}, index=level_tags[:-1]
    )
    print("Strong-column-weak-beam check (needs to be > 1.00)")
    print(scwb_check)
    print()

    if risk_category == "iv":
        ext_res = []
        int_res = []

        for place in ["exterior", "interior"]:
            for level_num in range(len(level_tags) - 1):
                this_level = level_tags[level_num]
                level_above = level_tags[level_num + 1]
                beam_sec = beam2_secs[this_level].properties
                # dist_a = beam_sec['bf'] * 0.625
                # dist_b = beam_sec['d'] * 0.75
                # sh = dist_a + dist_b/2.00
                sh = 0.00  # WUF-W connection
                if place == "exterior":
                    col_sec = col_ext2_secs[this_level].properties
                    col_sec_above = col_ext2_secs[level_above].properties
                else:
                    col_sec = col_int2_secs[this_level].properties
                    col_sec_above = col_int2_secs[level_above].properties
                if place == "exterior":
                    capacity = smrf_scwb(
                        col_sec,
                        col_sec_above,
                        beam_sec,
                        col_puc_int[place][this_level],
                        (
                            1.2 * beam_udls_dead[this_level]
                            + 0.50 * 0.40 * beam_udls_live[this_level]
                        ),
                        1.00,
                        hi_diff[level_num],
                        25.00 * 12.00,
                        None,
                        None,
                        None,
                        sh,
                        50000.00,
                    )
                    ext_res.append(capacity)
                if place == "interior":
                    capacity = smrf_scwb(
                        col_sec,
                        col_sec_above,
                        beam_sec,
                        col_puc_int[place][this_level],
                        (
                            beam_udls_dead[this_level]
                            + 0.50 * 0.40 * beam_udls_live[this_level]
                        ),
                        1.00,
                        hi_diff[level_num],
                        25.00 * 12.00,
                        beam_sec,
                        (
                            1.2 * beam_udls_dead[this_level]
                            + 0.50 * 0.40 * beam_udls_live[this_level]
                        ),
                        1.00,
                        sh,
                        50000.00,
                    )
                    int_res.append(capacity)
        scwb_check2 = pd.DataFrame(
            {"exterior": ext_res, "interior": int_res}, index=level_tags[:-1]
        )
        print("Strong-column-weak-beam check (needs to be > 1.00), interior frame")
        print(scwb_check2)
        print()

    if get_doubler_plates:
        # calculate doubler plate requirements
        ext_doubler_thickness = []
        int_doubler_thickness = []
        ext_doubler_ratio = []
        int_doubler_ratio = []
        for place in ["exterior", "interior"]:
            for level_num, _ in enumerate(level_tags):
                this_level = level_tags[level_num]
                beam_sec = beam_secs[this_level].properties
                sh = 0.00
                if place == "exterior":
                    col_sec = col_ext_secs[this_level].properties
                else:
                    col_sec = col_int_secs[this_level].properties
                if place == "interior":
                    tdoub = smrf_pz_doubler_plate_requirement(
                        col_sec,
                        beam_sec,
                        1.00,
                        25.00 * 12.00,
                        "interior",
                        sh,
                        50000.00,
                    )
                    # round to the nearest 1/16th of an inch
                    tdoub = np.ceil(tdoub / (1.00 / 16.00)) * (1.00 / 16.00)
                    if tdoub != 0.0:
                        tdoub = max(tdoub, 1.0 / 4.00)  # AISC 341-16 6e.3
                    int_doubler_thickness.append(tdoub)
                    int_doubler_ratio.append(tdoub / col_sec["tw"])
                    assert (
                        tdoub + col_sec["tw"]
                        > (col_sec["d"] + beam_sec["d"]) / 90.00
                    )
                else:
                    tdoub = smrf_pz_doubler_plate_requirement(
                        col_sec,
                        beam_sec,
                        1.00,
                        25.00 * 12.00,
                        "exterior",
                        sh,
                        50000.00,
                    )
                    # round to the nearest 1/16th of an inch
                    tdoub = np.ceil(tdoub / (1.00 / 16.00)) * (1.00 / 16.00)
                    if tdoub != 0.00:
                        tdoub = max(tdoub, 1.0 / 4.00)  # AISC 341-16 6e.3
                    ext_doubler_thickness.append(tdoub)
                    ext_doubler_ratio.append(tdoub / col_sec["tw"])
                    assert (
                        tdoub + col_sec["tw"]
                        > (col_sec["d"] + beam_sec["d"]) / 90.00
                    )
        pz_check = pd.DataFrame(
            {"exterior": ext_doubler_thickness, "interior": int_doubler_thickness},
            index=level_tags,
        )
        pz_ratio_check = pd.DataFrame(
            {"exterior": ext_doubler_ratio, "interior": int_doubler_ratio},
            index=level_tags,
        )
        print("Doubler Plate Thickness Requirement (in)")
        print(pz_check)
        print()
        print("Doubler Plate Thickness/Web thickness ratios")
        print(pz_ratio_check)
        print()

        if risk_category == "iv":
            ext_doubler_thickness = []
            int_doubler_thickness = []
            ext_doubler_ratio = []
            int_doubler_ratio = []
            for place in ["exterior", "interior"]:
                for level_num, _ in enumerate(level_tags):
                    this_level = level_tags[level_num]
                    beam_sec = beam2_secs[this_level].properties
                    sh = 0.00
                    if place == "exterior":
                        col_sec = col_ext2_secs[this_level].properties
                    else:
                        col_sec = col_int2_secs[this_level].properties
                    if place == "interior":
                        tdoub = smrf_pz_doubler_plate_requirement(
                            col_sec,
                            beam_sec,
                            1.00,
                            25.00 * 12.00,
                            "interior",
                            sh,
                            50000.00,
                        )
                        # round to the nearest 1/16th of an inch
                        tdoub = np.ceil(tdoub / (1.00 / 16.00)) * (1.00 / 16.00)
                        if tdoub != 0.0:
                            tdoub = max(tdoub, 1.0 / 4.00)  # AISC 341-16 6e.3
                        int_doubler_thickness.append(tdoub)
                        int_doubler_ratio.append(tdoub / col_sec["tw"])
                        assert (
                            tdoub + col_sec["tw"]
                            > (col_sec["d"] + beam_sec["d"]) / 90.00
                        )
                    else:
                        tdoub = smrf_pz_doubler_plate_requirement(
                            col_sec,
                            beam_sec,
                            1.00,
                            25.00 * 12.00,
                            "exterior",
                            sh,
                            50000.00,
                        )
                        # round to the nearest 1/16th of an inch
                        tdoub = np.ceil(tdoub / (1.00 / 16.00)) * (1.00 / 16.00)
                        if tdoub != 0.00:
                            tdoub = max(tdoub, 1.0 / 4.00)  # AISC 341-16 6e.3
                        ext_doubler_thickness.append(tdoub)
                        ext_doubler_ratio.append(tdoub / col_sec["tw"])
                        assert (
                            tdoub + col_sec["tw"]
                            > (col_sec["d"] + beam_sec["d"]) / 90.00
                        )
            pz_check = pd.DataFrame(
                {
                    "exterior": ext_doubler_thickness,
                    "interior": int_doubler_thickness,
                },
                index=level_tags,
            )
            pz_ratio_check = pd.DataFrame(
                {"exterior": ext_doubler_ratio, "interior": int_doubler_ratio},
                index=level_tags,
            )
            print("Inner frame results:")
            print()
            print("Doubler Plate Thickness Requirement (in)")
            print(pz_check)
            print()
            print("Doubler Plate Thickness/Web thickness ratios")
            print(pz_ratio_check)
            print()

        if risk_category == "iv":
            ext_doubler_thickness = []
            int_doubler_thickness = []
            ext_doubler_ratio = []
            int_doubler_ratio = []
            for place in ["exterior", "interior"]:
                for level_num, _ in enumerate(level_tags):
                    this_level = level_tags[level_num]
                    beam_sec = beam2_secs[this_level].properties
                    sh = 0.00
                    if place == "exterior":
                        col_sec = col_ext2_secs[this_level].properties
                    else:
                        col_sec = col_int2_secs[this_level].properties
                    if place == "interior":
                        tdoub = smrf_pz_doubler_plate_requirement(
                            col_sec,
                            beam_sec,
                            1.00,
                            25.00 * 12.00,
                            "interior",
                            sh,
                            50000.00,
                        )
                        # round to the nearest 1/16th of an inch
                        tdoub = np.ceil(tdoub / (1.00 / 16.00)) * (1.00 / 16.00)
                        if tdoub != 0.0:
                            tdoub = max(tdoub, 1.0 / 4.00)  # AISC 341-16 6e.3
                        int_doubler_thickness.append(tdoub)
                        int_doubler_ratio.append(tdoub / col_sec["tw"])
                        assert (
                            tdoub + col_sec["tw"]
                            > (col_sec["d"] + beam_sec["d"]) / 90.00
                        )
                    else:
                        tdoub = smrf_pz_doubler_plate_requirement(
                            col_sec,
                            beam_sec,
                            1.00,
                            25.00 * 12.00,
                            "exterior",
                            sh,
                            50000.00,
                        )
                        # round to the nearest 1/16th of an inch
                        tdoub = np.ceil(tdoub / (1.00 / 16.00)) * (1.00 / 16.00)
                        if tdoub != 0.00:
                            tdoub = max(tdoub, 1.0 / 4.00)  # AISC 341-16 6e.3
                        ext_doubler_thickness.append(tdoub)
                        ext_doubler_ratio.append(tdoub / col_sec["tw"])
                        assert (
                            tdoub + col_sec["tw"]
                            > (col_sec["d"] + beam_sec["d"]) / 90.00
                        )
            pz_check = pd.DataFrame(
                {
                    "exterior": ext_doubler_thickness,
                    "interior": int_doubler_thickness,
                },
                index=level_tags,
            )
            pz_ratio_check = pd.DataFrame(
                {"exterior": ext_doubler_ratio, "interior": int_doubler_ratio},
                index=level_tags,
            )
            print("Doubler Plate Thickness Requirement (in)")
            print(pz_check)
            print()
            print("Doubler Plate Thickness/Web thickness ratios")
            print(pz_ratio_check)
            print()

    # check beam strength
    capacity_ratios = []
    if get_beam_checks:
        for level_counter in range(1, 3 + 1):
            level_tag = "level_" + str(level_counter)
            for xpt in x_locs[:-1]:
                xpt += 30.00
                comp = query.retrieve_component(xpt, 0.00, level_counter)
                assert comp is not None
                for elm in comp.elements.values():
                    sec = elm.section.properties
                    # rbs_proportion = 0.60
                    rbs_proportion = 1.00
                    c_rbs_j = sec["bf"] * (1.0 - rbs_proportion) / 2.0
                    z_rbs_j = sec["Zx"] - 2.0 * c_rbs_j * sec["tf"] * (
                        sec["d"] - sec["tf"]
                    )
                    fy = 50000.00
                    m_pr = fy * z_rbs_j
                    factor_i = (
                        abs(strength_combo.envelope_basic_forces(elm, 2)[0]["mz"])
                        / m_pr
                    )
                    factor_j = (
                        abs(strength_combo.envelope_basic_forces(elm, 2)[1]["mz"])
                        / m_pr
                    )
                    factor = pd.concat([factor_i, factor_j]).max()
                    capacity_ratios.append(factor)

    print("Beam strength check")
    print([f"{cap:.2f}" for cap in capacity_ratios])
    print()

    # ~~~~~~~~~~~~ #
    # print output #
    # ~~~~~~~~~~~~ #

    msgs: str = "\n"

    def get_warning(value, must_be, other, msg):
        """
        Return a warning message if applicable
        """
        res = ""
        if must_be == "<":
            if value > other:
                res = f"Warning: {msg}. {value:.3f} > {other:.3f}\n"
            else:
                res = ""
        elif must_be == ">":
            if value < other:
                res = f"Warning: {msg}. {value:.3f} < {other:.3f}\n"
            else:
                res = ""
        return res

    weight = 0.00
    for elm in mdl.list_of_elements():
        if not hasattr(elm, "clear_length"):
            continue
        length = elm.clear_length()
        weight += elm.section.weight_per_length() * length

    # drift constraint
    for lvl_idx in range(1, num_lvls + 1):
        msgs += get_warning(dr[lvl_idx] / max_drift, "<", 1.00, "drift")

    # strong column weak beam check
    for val in ext_res + int_res:
        msgs += get_warning(val, ">", 1.00, "SCWB")

    # stability check
    for i in range(len(p_nodes)):
        if thetas[i] / theta_lim > 1.00:
            msgs += get_warning(
                thetas[i], "<", theta_lim, f"Stability Problem @ lvl {i+1}"
            )

    print()
    print("Drift capacity ratios, X direction (MODAL):")
    print()
    print([f"{cap:.3f}" for cap in [drift / max_drift for drift in dr.values()]])
    print()
    print(f"  periods: {ts}")
    print(f"  steel weight:  {weight:.2f}")
    print(f"  warnings: {msgs}")
