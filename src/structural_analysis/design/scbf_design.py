"""
Design a n-story SCBF system
"""

import copy
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
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass
from osmg.postprocessing.design import LoadCombination
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d  # type: ignore
from scipy.optimize import fsolve

# from osmg.graphics.preprocessing_3d import show
# from osmg.graphics.postprocessing_3d import show_deformed_shape
# from osmg.graphics.postprocessing_3d import show_basic_forces
# from osmg.graphics.postprocessing_3d import show_basic_forces_combo

# pylint: disable=invalid-name
# pylint: disable=too-many-lines
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-locals

nparr = npt.NDArray[np.float64]


def design_scbf_lrrs(
    num_lvls,
    beams: dict[int, list[str]],
    columns: dict[int, list[str]],
    braces: dict[int, list[str]],
    coeff: list[int],
    beam_udls_dead: dict[str, float],
    beam_udls_live: dict[str, float],
    lvl_weight: dict[str, float],
    design_params: dict[str, float],
    site_characteristics: dict[str, float],
    tmax_params: dict[str, float],
    mlp_periods: nparr,
    mlp_des_spc: nparr,
    num_braces: int,
    show_metadata=False,
):
    """
    Design a n-story SCBF system

    Args:
      num_braces: How many lines of braced bays exist in the plan
      (not # of individual braces)

    """

    # Stage: Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # brace design parameters
    steel_e = 29000.0  # ksi
    f_y_br = 42.00  # ksi, value is for round HSSs
    r_y_br = 1.4
    f_u_br = f_y_br * r_y_br  # ksi
    f_y_fr = 50.00  # ksi
    f_y_gu = 50.00  # ksi
    f_u_gu = 65.00  # ksi
    r_y_gu = 1.1
    f_exx_weld = 70.00  # ksi
    n_weld = 4.00

    # Stage: Model definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    beam_coeff_lvl: dict[int, int] = {}
    col_coeff_lvl: dict[int, int] = {}
    brace_coeff_lvl: dict[int, int] = {}
    running_idx = 0
    for lvl_idx in range(1, num_lvls + 1):
        beam_coeff_lvl[lvl_idx] = running_idx
        running_idx += 1
    for lvl_idx in range(1, num_lvls + 1):
        col_coeff_lvl[lvl_idx] = running_idx
        running_idx += 1
    for lvl_idx in range(1, num_lvls + 1):
        brace_coeff_lvl[lvl_idx] = running_idx
        running_idx += 1

    mdl = Model("design_model")
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
    defaults.load_util_rigid_elastic(mdl)

    def section_from_index(indx, list_of_section_names, sectype):
        """
        Defines a section from the given index
        """
        secg.load_aisc_from_database(
            sectype,
            [list_of_section_names[indx]],
            "default steel",
            "default steel",
            ElasticSection,
        )
        res_sec = mdl.elastic_sections.retrieve_by_attr(
            "name", list_of_section_names[indx]
        )
        return res_sec

    beam_secs: dict = {}
    col_secs: dict = {}
    brace_secs: dict = {}
    for lvl_idx in range(1, num_lvls + 1):
        beam_secs[f"level_{lvl_idx}"] = section_from_index(
            coeff[beam_coeff_lvl[lvl_idx]], beams[lvl_idx], "W"
        )
        col_secs[f"level_{lvl_idx}"] = section_from_index(
            coeff[col_coeff_lvl[lvl_idx]], columns[lvl_idx], "W"
        )
        brace_secs[f"level_{lvl_idx}"] = section_from_index(
            coeff[brace_coeff_lvl[lvl_idx]], braces[lvl_idx], "HSS_circ"
        )

    print("Sections")
    print("Beams")
    print([sec.name for sec in beam_secs.values()])
    print("Columns")
    print([sec.name for sec in col_secs.values()])
    print("Braces")
    print([sec.name for sec in brace_secs.values()])
    print()

    # define structural elements
    x_locs = np.array([0.00, 25.00, 50.00]) * 12.00  # (in)

    for level_counter in range(num_lvls):
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])
        for xpt in x_locs:
            sec = col_secs[level_tag]
            pt = np.array((xpt, 0.00))
            # mcg.add_pz_active(
            #     pt[0], pt[1],
            #     sec,
            #     steel_phys_mat,
            #     np.pi / 2.00,
            #     sec.properties['d'],
            #     beam_secs[level_tag].properties['d'],
            #     "steel_w_col_pz_updated",
            #     {
            #      'pz_doubler_plate_thickness': 0.00,
            #      'axial_load_ratio': 0.00,
            #      'consider_composite': False,
            #      'slab_depth': 0.00,
            #      'location': 'interior',
            #      'only_elastic': True,
            #     }
            # )
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
                method="generate_hinged_component_assembly",
                additional_args={
                    "zerolength_gen_i": None,
                    "zerolength_gen_args_i": {},
                    "zerolength_gen_j": release_6,
                    "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
                },
            )
        for ipt_idx in range(len(x_locs) - 1):
            pt_i = np.array((x_locs[ipt_idx], 0.00))
            pt_j = np.array((x_locs[ipt_idx + 1], 0.00))
            sec = beam_secs[level_tag]
            mcg.add_horizontal_active(
                pt_i[0],
                pt_i[1],
                pt_j[0],
                pt_j[1],
                np.array((0.0, 0.0, 0.0)),
                np.array((0.0, 0.0, 0.0)),
                # 'middle_back',
                # 'middle_front',
                "centroid",
                "centroid",
                "Linear",
                1,
                sec,
                ElasticBeamColumn,
                "top_center",
                # 'centroid',
                method="generate_hinged_component_assembly",
                additional_args={
                    "zerolength_gen_i": release_6,
                    "zerolength_gen_args_i": {"distance": 1.00, "n_sub": 1},
                    "zerolength_gen_j": release_6,
                    "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
                },
            )

    vertical_offsets = [-beam_secs["level_1"].properties["d"] / 2.00]
    for level_counter in range(num_lvls):
        level_tag = "level_" + str(level_counter + 1)
        vertical_offsets.append(-beam_secs[level_tag].properties["d"] / 2.00)

    # before adding the braces, create a copy of the existing model
    # for later use in capacity design
    mdl_bareframe = copy.deepcopy(mdl)
    # from osmg.graphics.preprocessing_3d import show
    # show(mdl_bareframe)
    # note: no preprocessing or loadcases so far, just the model with some
    # elements, which is what we want.

    # continuing with element definition in the original model

    # left braces
    model_braces = []
    for level_counter in range(num_lvls):
        level_tag = "level_" + str(level_counter + 1)
        sec = brace_secs[level_tag]
        mdl.levels.set_active([level_counter + 1])
        if level_counter % 2 == 0:
            pt_A = np.array((0.00, 0.00))
            pt_B = np.array((25.00 * 12.00, 0.00))
        else:
            pt_A = np.array((25.00 * 12.00, 0.00))
            pt_B = np.array((0.00, 0.00))
        added_braces = mcg.add_diagonal_active(
            pt_B[0],
            pt_B[1],
            pt_A[0],
            pt_A[1],
            np.array((0.00, 0.00, vertical_offsets[level_counter + 1])),
            np.array((0.00, 0.00, vertical_offsets[level_counter])),
            # np.array((0.00, 0.00, 0.00)),
            # np.array((0.00, 0.00, 0.00)),
            "centroid",
            "centroid",
            "Linear",
            1,
            sec,
            ElasticBeamColumn,
            "centroid",
            method="generate_hinged_component_assembly",
            additional_args={
                "zerolength_gen_i": release_6,
                "zerolength_gen_args_i": {"distance": 1.00, "n_sub": 1},
                "zerolength_gen_j": release_6,
                "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
            },
        )
        model_braces.append(list(added_braces.values())[0])

    # right braces
    for level_counter in range(num_lvls):
        level_tag = "level_" + str(level_counter + 1)
        sec = brace_secs[level_tag]
        mdl.levels.set_active([level_counter + 1])
        if level_counter % 2 == 0:
            pt_A = np.array((25.00 * 12.00, 0.00))
            pt_B = np.array((50.00 * 12.00, 0.00))
        else:
            pt_A = np.array((50.00 * 12.00, 0.00))
            pt_B = np.array((25.00 * 12.00, 0.00))
        mcg.add_diagonal_active(
            pt_A[0],
            pt_A[1],
            pt_B[0],
            pt_B[1],
            np.array((0.00, 0.00, vertical_offsets[level_counter + 1])),
            np.array((0.00, 0.00, vertical_offsets[level_counter])),
            # np.array((0.00, 0.00, 0.00)),
            # np.array((0.00, 0.00, 0.00)),
            "centroid",
            "centroid",
            "Linear",
            1,
            sec,
            ElasticBeamColumn,
            "centroid",
            method="generate_hinged_component_assembly",
            additional_args={
                "zerolength_gen_i": release_6,
                "zerolength_gen_args_i": {"distance": 1.00, "n_sub": 1},
                "zerolength_gen_j": release_6,
                "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
            },
        )

    # leaning column
    for level_counter in range(num_lvls):
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])

        sec = beam_secs[level_tag]
        d = sec.properties["d"]

        pt = np.array((75.00 * 12.00, 0.00))
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
                "zerolength_gen_i": None,
                "zerolength_gen_args_i": {},
                "zerolength_gen_j": release_6,
                "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
            },
        )
        pt_i = np.array((x_locs[-1], 0.00))
        pt_j = pt
        mcg.add_horizontal_active(
            pt_i[0],
            pt_i[1],
            pt_j[0],
            pt_j[1],
            np.array((0.0, 0.0, -d / 2.0)),
            np.array((0.0, 0.0, -d / 2.0)),
            "centroid",
            "centroid",
            "Linear",
            1,
            rigidsec,
            ElasticBeamColumn,
            "centroid",
            method="generate_hinged_component_assembly",
            additional_args={
                "zerolength_gen_i": release_6,
                "zerolength_gen_args_i": {"distance": 1.00, "n_sub": 1},
                "zerolength_gen_j": release_6,
                "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
            },
        )
    for level_counter in range(num_lvls):
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])

        sec = beam_secs[level_tag]
        d = sec.properties["d"]

        pt = np.array((-25.00 * 12.00, 0.00))
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
                "zerolength_gen_i": None,
                "zerolength_gen_args_i": {},
                "zerolength_gen_j": release_6,
                "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
            },
        )
        pt_i = np.array((x_locs[0], 0.00))
        pt_j = pt
        mcg.add_horizontal_active(
            pt_i[0],
            pt_i[1],
            pt_j[0],
            pt_j[1],
            np.array((0.0, 0.0, -d / 2.0)),
            np.array((0.0, 0.0, -d / 2.0)),
            "centroid",
            "centroid",
            "Linear",
            1,
            rigidsec,
            ElasticBeamColumn,
            "centroid",
            method="generate_hinged_component_assembly",
            additional_args={
                "zerolength_gen_i": release_6,
                "zerolength_gen_args_i": {"distance": 1.00, "n_sub": 1},
                "zerolength_gen_j": release_6,
                "zerolength_gen_args_j": {"distance": 1.00, "n_sub": 1},
            },
        )

    # from osmg.graphics.preprocessing_3d import show
    # show(mdl, extrude=True)

    p_nodes_i = []
    p_nodes_j = []
    for i in range(1, num_lvls + 1):
        ni = query.search_node_lvl(-25.00 * 12.00, 0.00, i)
        assert ni
        p_nodes_i.append(ni)
        nj = query.search_node_lvl(75.00 * 12.00, 0.00, i)
        assert nj
        p_nodes_j.append(nj)

    # restrict motion in XZ plane
    for node in mdl.list_of_all_nodes():
        node.restraint = [False, True, False, True, False, True]
    # fix base
    for node in mdl.levels[0].nodes.values():
        node.restraint = [True] * 6

    # subset model for plots (without leaning column)
    subset_model = mdl.initialize_empty_copy("subset_1")
    fudge = 50.00
    mdl.transfer_by_polygon_selection(
        subset_model,
        np.array(
            (
                (-fudge, -fudge),
                ((50.00 * 12.00) + fudge, -fudge),
                ((50.00 * 12.00) + fudge, +fudge),
                (-fudge, +fudge),
            )
        ),
    )

    # from osmg.graphics.preprocessing_3d import show
    # show(subset_model)

    # assign loads

    lc_dead = LoadCase("dead", mdl)
    self_mass(mdl, lc_dead)
    self_weight(mdl, lc_dead)
    lc_live = LoadCase("live", mdl)

    for level_counter in range(1, num_lvls + 1):
        level_tag = "level_" + str(level_counter)
        for xpt in [20.00 * 12.00, 30.00 * 12.00]:
            comp = query.retrieve_component(xpt, 0.00, level_counter)
            assert comp is not None
            for elm in comp.elements.values():
                if not isinstance(elm, ElasticBeamColumn):
                    continue
                lc_dead.line_element_udl[elm.uid].add_glob(
                    np.array((0.00, 0.00, -beam_udls_dead[level_tag]))
                )
                lc_live.line_element_udl[elm.uid].add_glob(
                    np.array((0.00, 0.00, -beam_udls_live[level_tag]))
                )
        nd = query.search_node_lvl(-25.00 * 12.00, 0.00, level_counter)
        assert nd is not None
        lc_dead.node_loads[nd.uid].val += np.array(
            (
                0.00,
                0.00,
                -lvl_weight[level_tag] / 2.00 / num_braces,
                0.00,
                0.00,
                0.00,
            )
        )
        mass = lvl_weight[level_tag] / common.G_CONST_IMPERIAL / 2.00 / num_braces
        lc_dead.node_mass[nd.uid].val += np.array(
            (mass, mass, mass, 0.00, 0.00, 0.00)
        )
        nd = query.search_node_lvl(75.00 * 12.00, 0.00, level_counter)
        assert nd is not None
        lc_dead.node_loads[nd.uid].val += np.array(
            (
                0.00,
                0.00,
                -lvl_weight[level_tag] / 2.00 / num_braces,
                0.00,
                0.00,
                0.00,
            )
        )
        mass = lvl_weight[level_tag] / common.G_CONST_IMPERIAL / 2.00 / num_braces
        lc_dead.node_mass[nd.uid].val += np.array(
            (mass, mass, mass, 0.00, 0.00, 0.00)
        )

    # from osmg.graphics.preprocessing_3d import show
    # show(subset_model, lc_dead, extrude=True)

    # Stage: Running all necessary analyses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # earthquake - ELF
    # design parameters
    c_d = design_params["Cd"]
    r_factor = design_params["R"]
    i_e = design_params["Ie"]
    ecc_ampl = design_params["ecc_ampl"]
    max_drift = design_params["max_drift"]

    # site characteristics
    s_ds = site_characteristics["Sds"]
    s_d1 = site_characteristics["Sd1"]

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

    def t_max(c_t, expnt, height, s_d1):
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

    # ELF analysis: for the global stability check

    # period estimation (Table 12.8-2)
    ct = tmax_params["ct"]
    exponent = tmax_params["exponent"]
    T_max = t_max(ct, exponent, hi[-1] / 12.00, s_d1)

    print(f"T_max = {T_max:.2f} s\n")

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
    print("Modal periods:")
    print([f"{t:.2f}" for t in ts])
    print()

    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # from osmg.graphics.postprocessing_3d import show_basic_forces
    # show_deformed_shape(
    #     modal_analysis, 'modal', 0, 0.00,
    #     False, subset_model=subset_model
    # )
    # show_basic_forces(
    #     modal_analysis, 'modal', 0, 2.00, 0.00, 0.00,
    #     0.00, 0.00, 10, 1.00, 1.00,
    #     global_axes=True, subset_model=subset_model)
    # show_deformed_shape(
    #     modal_analysis, 'modal', 0, 0.00,
    #     False
    # )
    # show_basic_forces(
    #     modal_analysis, 'modal', 0, 2.00, 0.00, 0.00,
    #     0.00, 0.00, 10, 1.00, 1.00,
    #     global_axes=True)

    # mode shape
    disps = np.zeros(len(p_nodes_i))
    for i, p_node in enumerate(p_nodes_i):
        assert p_node
        disps[i] = modal_analysis.results["modal"].node_displacements[p_node.uid][0][
            0
        ]
    disps /= disps[-1]
    print("Mode Shape: ")
    print([f"{d:.3f}" for d in disps])
    print()

    print(f"T_modal = {ts[0]:.2f} s\n")

    t_use = min(ts[0], T_max)
    wi = np.array(list(lvl_weight.values())) / num_braces
    vb_elf = np.sum(wi) * cs(t_use, s_ds, s_d1, r_factor, i_e) * ecc_ampl
    print(f"Seismic weight: {np.sum(wi)/1000.00:.0f} kips")
    print(f"V_b_elf = {vb_elf/1000:.2f} kips \n")
    print(f"Cs = {cs(t_use, s_ds, s_d1, r_factor, i_e):.3f}")
    cvx = wi * hi ** k(ts[1]) / np.sum(wi * hi ** k(ts[1]))
    fx = vb_elf * cvx

    lc_elf = LoadCase("elf", mdl)
    for i, nd in enumerate(p_nodes_i):
        assert nd is not None
        lc_elf.node_loads[nd.uid].add(
            np.array((fx[i] / 2.00, 0.00, 0.00, 0.00, 0.00, 0.00))
        )
    for i, nd in enumerate(p_nodes_j):
        assert nd is not None
        lc_elf.node_loads[nd.uid].add(
            np.array((fx[i] / 2.00, 0.00, 0.00, 0.00, 0.00, 0.00))
        )

    elf_anl = solver.StaticAnalysis(mdl, {"elf": lc_elf})
    elf_anl.run()
    elf_disps = []
    for pnode in p_nodes_i:
        elf_disps.append(elf_anl.results["elf"].node_displacements[pnode.uid][0][0])

    elf_combo = LoadCombination(
        mdl, {"+E": [(1.00, elf_anl, "elf")], "-E": [(-1.00, elf_anl, "elf")]}
    )

    # from osmg.graphics.postprocessing_3d import show_basic_forces_combo
    # show_basic_forces_combo(
    #     elf_combo, 1.00, .00, .0, .0, .0, 50, global_axes=True,
    #     force_conversion=1.00/1000.00,
    #     moment_conversion=1.00/12.00/1000.00,
    # )

    # design combinations

    static_anl = solver.StaticAnalysis(mdl, {"dead": lc_dead, "live": lc_live})
    static_anl.run()

    # show_deformed_shape(
    #     static_anl, 'live', 0, 0.00, False, subset_model=subset_model)

    # show_basic_forces(
    #     static_anl, 'dead', 0, 1.00, 0.00, 0.00,
    #     1.00e-1, 0.00, 10,
    #     global_axes=True, subset_model=subset_model)

    rsa = solver.ModalResponseSpectrumAnalysis(
        mdl, lc_modal, num_modes, mlp_periods, mlp_des_spc, "x"
    )
    rsa.run()
    assert rsa.anl is not None
    assert rsa.anl.results is not None
    ts = rsa.anl.results["modal"].periods

    assert rsa is not None
    assert rsa.vb_modal is not None
    vb_modal = np.sqrt(np.sum(rsa.vb_modal**2)) / 1000 / (r_factor / i_e) * ecc_ampl
    print(f"V_b_modal = {vb_modal:.2f} kips \n")

    if ts[0] > T_max:
        vb_ampl = max(1.00, (vb_elf / 1000.00) / vb_modal)
    else:
        vb_ampl = 1.00

    drift_combo = LoadCombination(
        mdl,
        {
            "D+L+E": [
                (1.00, static_anl, "dead"),
                (1.00, static_anl, "live"),
                (1.00 / (r_factor / i_e) * ecc_ampl, rsa, "modal"),
            ],
            "D+L-E": [
                (1.00, static_anl, "dead"),
                (1.00, static_anl, "live"),
                (-1.00 / (r_factor / i_e) * ecc_ampl, rsa, "modal"),
            ],
        },
    )  # type: ignore # indexs from ASCE 7-22 12.8.6.1

    comb_1 = LoadCombination(
        mdl, {"1.4D": [(1.40, static_anl, "dead")]}
    )  # type: ignore
    comb_2 = LoadCombination(
        mdl, {"1.2D+1.6L": [(1.20, static_anl, "dead"), (1.60, static_anl, "live")]}
    )  # type: ignore
    comb_3 = LoadCombination(
        mdl,
        {
            "1.2(D+0.2Sds)+L+E": [
                (1.20 + 0.20 * s_ds, static_anl, "dead"),
                (1.00, static_anl, "live"),
                (1.00 / (r_factor / i_e) * ecc_ampl * vb_ampl, rsa, "modal"),
            ],
            "1.2(D+0.2Sds)+L-E": [
                (1.20 + 0.20 * s_ds, static_anl, "dead"),
                (1.00, static_anl, "live"),
                (-1.00 / (r_factor / i_e) * ecc_ampl * vb_ampl, rsa, "modal"),
            ],
        },
    )  # type: ignore
    comb_4 = LoadCombination(
        mdl,
        {
            "0.9(D-0.2Sds)+L+E": [
                (0.90 - 0.20 * s_ds, static_anl, "dead"),
                (1.00, static_anl, "live"),
                (1.00 / (r_factor / i_e) * ecc_ampl * vb_ampl, rsa, "modal"),
            ],
            "0.9(D-0.2Sds)+L-E": [
                (0.90 - 0.20 * s_ds, static_anl, "dead"),
                (1.00, static_anl, "live"),
                (-1.00 / (r_factor / i_e) * ecc_ampl * vb_ampl, rsa, "modal"),
            ],
        },
    )  # type: ignore

    design_combos = [comb_1, comb_2, comb_3, comb_4]

    # from osmg.graphics.postprocessing_3d import show_basic_forces_combo
    # show_basic_forces_combo(
    #     comb_1, 1.00, 0.00, 0.00, 0.00, 0.00,
    #     30, 1.00/1000.00, 1.00/1000.00/12.00, True, subset_model=subset_model
    # )
    # show_basic_forces_combo(
    #     comb_2, 1.00, 0.00, 0.00, 0.00, 0.00,
    #     30, 1.00/1000.00, 1.00/1000.00/12.00, True, subset_model=subset_model
    # )
    # show_basic_forces_combo(
    #     comb_3, 1.00, 0.00, 0.00, 0.00, 0.00,
    #     30, 1.00/1000.00, 1.00/1000.00/12.00, True, subset_model=subset_model
    # )
    # show_basic_forces_combo(
    #     comb_4, 1.00, 0.00, 0.00, 0.00, 0.00,
    #     30, 1.00/1000.00, 1.00/1000.00/12.00, True, subset_model=subset_model
    # )

    # Stage: Global checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # construct and issue warnings
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

    # Global stability (P-Delta Effects). ASCE 7-22 12.8.7
    # units used here: lb, in
    thetas = np.zeros(len(p_nodes_i))
    theta_lim = 0.10
    lvlw = np.array(list(lvl_weight.values())) / num_braces
    for lvl_idx in range(len(p_nodes_i)):
        if lvl_idx == 0:
            deltax = np.max(
                np.abs(
                    [
                        r[0]
                        for r in (
                            elf_combo.envelope_node_displacement(p_nodes_i[lvl_idx])
                        )
                    ]
                )
            )
        else:
            deltax = np.max(
                np.abs(
                    [
                        r[0]
                        for r in (
                            elf_combo.envelope_node_displacement_diff(
                                p_nodes_i[lvl_idx], p_nodes_i[lvl_idx - 1]
                            )
                        )
                    ]
                )
            )
        px = np.sum(lvlw[lvl_idx:])
        vx = np.sum(fx[lvl_idx:])
        hsx = hi[lvl_idx]
        thetas[lvl_idx] = (px / hsx) / (vx / deltax)
    print("P-Delta capacity ratios")
    print([f"{t:.2f}" for t in thetas / theta_lim])  # should be < 1
    print()

    for lvl_idx in range(num_lvls):
        msgs += get_warning(thetas[lvl_idx], "<", 1.00, "P-Delta check")

    # drift limits
    drifts = []
    for i in range(num_lvls):
        if i == 0:
            drifts.append(
                np.max(
                    np.abs(
                        [
                            r[0]
                            for r in (
                                drift_combo.envelope_node_displacement(p_nodes_i[i])
                            )
                        ]
                    )
                )
                / hi_diff[i]
                * c_d
                / i_e
            )
        else:
            drifts.append(
                np.max(
                    np.abs(
                        [
                            r[0]
                            for r in (
                                drift_combo.envelope_node_displacement_diff(
                                    p_nodes_i[i], p_nodes_i[i - 1]
                                )
                            )
                        ]
                    )
                )
                / hi_diff[i]
                * c_d
                / i_e
            )
    print("Drift Capacity Ratios")
    print([f"{d/max_drift:.2f}" for d in drifts])
    print()

    for i in range(num_lvls):
        msgs += get_warning(
            drifts[i] / max_drift, "<", 1.00, "Drift Capacity Ratios"
        )

    # Stage: Member checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # brace connection design
    # level tag, value
    hinge_dist_bot: dict[int, float] = {}
    hinge_dist_top: dict[int, float] = {}
    pl_a_top: dict[int, float] = {}
    pl_a_bot: dict[int, float] = {}
    pl_b_top: dict[int, float] = {}
    pl_b_bot: dict[int, float] = {}
    p_ncs: dict[int, float] = {}
    p_uts: dict[int, float] = {}
    brace_angles: dict[int, float] = {}
    gusset_t_p: dict[int, float] = {}
    gusset_avg_buckl_len_bot: dict[int, float] = {}
    gusset_avg_buckl_len_top: dict[int, float] = {}
    brace_buckling_length: dict[int, float] = {}
    brace_l_c: dict[int, float] = {}

    def calc_gusset_plate_geometry(
        pl_a, t_p, brace_ang, col_depth, beam_depth, brace_depth
    ):
        """
        Calculates geometric properties associated with a gusset plate
        with a 8t_p elliptical clearence for the connecting brace.
        """
        pl_b = pl_a * np.tan(brace_ang)
        pl_a_pr = pl_a - col_depth / 2.00 - 8.00 * t_p
        pl_b_pr = pl_b - beam_depth / 2.00 - 8.00 * t_p
        pl_p = pl_a / pl_b
        x_pr = (1.00 / pl_a_pr**2 + (np.tan(brace_ang) / pl_b_pr) ** 2) ** (-0.50)
        y_pr = pl_b_pr * np.sqrt(1.00 - (x_pr / pl_a_pr) ** 2)
        beta_ang = np.arctan(-2.00 / pl_p * np.sqrt(pl_a_pr**2 / x_pr**2))
        cor = brace_depth / 2.00 * np.sin(beta_ang) * np.cos(brace_ang)
        l_pr = np.sqrt(x_pr**2 + y_pr**2) + cor
        # deterime how much the length is reduced due to the gusset plate
        # corner cutoff
        assert 0 <= brace_ang <= np.pi / 2.00
        if brace_ang > np.pi / 4.00:
            brace_ang = np.pi / 2.00 - brace_ang
        s = np.sin(np.pi / 2.00 - brace_ang) / np.sin(brace_ang) * brace_depth / 2.00
        hinge_dist = np.sqrt(pl_a**2 + pl_b**2) - np.sqrt(x_pr**2 + y_pr**2)
        l_c_avail = l_pr - s
        # average gusset plate buckling length
        # l2 based on beam geometry
        dist_b = beam_depth / 2.00 / np.tan(brace_ang)
        dist_c = np.sqrt((beam_depth / 2.00) ** 2 + dist_b**2)
        dist_diag = np.sqrt(pl_b**2 + pl_a**2)
        dist_l2_beam = dist_diag - dist_c - l_pr
        # l2 based on column geometry
        dist_b = (col_depth / 2.00) * np.tan(brace_ang)
        dist_c = np.sqrt((col_depth / 2.00) ** 2 + dist_b**2)
        dist_l2_col = dist_diag - dist_c - l_pr
        dist_l2 = min(dist_l2_beam, dist_l2_col)
        # l1 based on column geometry (we assume beam does not control)
        dist_h = (l_pr - s) * np.tan(30.00 / 180.00 * np.pi) + brace_depth / 2.00
        dist_b = np.tan(brace_ang) * dist_h
        dist_e = col_depth / 2.00 * np.tan(brace_ang)
        dist_f = np.sqrt((col_depth / 2.00) ** 2 + dist_e**2)
        dist_l1 = dist_diag - l_pr - dist_b - dist_f
        # l3 based on beam geometry (we assume column does not control)
        ang_b = np.pi / 2.00 - brace_ang
        ang_c = np.pi / 2.00 - ang_b
        dist_g = dist_h / np.sin(ang_c) * np.sin(ang_b)
        dist_w = beam_depth / 2.00 / np.tan(brace_ang)
        dist_k = np.sqrt((beam_depth / 2.00) ** 2 + dist_w**2)
        dist_l3 = dist_diag - l_pr - dist_g - dist_k
        dist_avg = (dist_l1 + dist_l2 + dist_l3) / 3.00
        return l_c_avail, hinge_dist, dist_avg

    def optim_gusset_plate_geometry(
        x, l_c, t_p, brace_ang, col_depth, beam_depth, brace_depth
    ):
        if x < 0:  # constrain to positive solutions
            return -x + 1.00
        a = calc_gusset_plate_geometry(
            x, t_p, brace_ang, col_depth, beam_depth, brace_depth
        )[0]
        res = a - l_c
        return res

    for level_counter in range(1, num_lvls + 1):
        level_tag = "level_" + str(level_counter)
        if level_counter == 1:
            prev_tag = "level_" + str(level_counter)
        else:
            prev_tag = "level_" + str(level_counter - 1)
        sec = brace_secs[level_tag]
        pr = sec.properties
        pr_col = col_secs[level_tag].properties
        pr_bm_top = beam_secs[level_tag].properties
        pr_bm_bot = beam_secs[prev_tag].properties
        # expected brace yield capacity
        p_ut = r_y_br * f_y_br * pr["A"]  # kips
        # brace-to-gusset connection length
        t_weld = pr["tdes"]
        beta = 0.75
        l_c = p_ut / (beta * 0.60 * f_exx_weld * n_weld * 0.707 * t_weld)  # in
        brace_l_c[level_counter] = l_c
        # brace base material check
        assert 0.75 * 0.60 * f_u_br * n_weld * l_c * t_weld > p_ut
        # determine plate thickness
        b_w = pr["OD"] + 2.00 * np.tan(30.00 / 360.00 * 2.00 * np.pi) * l_c  # in
        t_p = max(
            p_ut / (1.00 * r_y_gu * f_y_gu * b_w),  # yield
            p_ut / (0.85 * f_u_gu * b_w),  # tensile rupture
            p_ut
            / (
                0.85
                * (
                    0.60 * f_u_gu * pr["OD"] * pr["tdes"] * 2.00
                    + 1.00 * f_u_gu * pr["OD"]
                )
            ),  # block shear
        )  # in
        # round up to nearest 16th of an inch
        t_p = np.ceil(t_p * 16.00) / 16.00
        gusset_t_p[level_counter] = t_p
        # plate size determination
        brace_angle = np.arctan(
            (
                hi_diff[level_counter - 1]
                - pr_bm_bot["d"] / 2.00
                + pr_bm_top["d"] / 2.00
            )
            / (25.00 * 12.00)
        )
        brace_angles[level_counter] = brace_angle

        trial_pl_a = np.floor(l_c * 4.00)
        pl_a = fsolve(
            optim_gusset_plate_geometry,
            trial_pl_a,
            args=(l_c, t_p, brace_angle, pr_col["d"], pr_bm_bot["d"], pr["OD"]),
        )
        assert len(pl_a) == 1
        pl_a_val = pl_a[0]
        # round up to the nearest 1/16 of an inch
        pl_a_val = np.ceil(pl_a_val * 16.00) / 16.00
        res = calc_gusset_plate_geometry(
            pl_a_val, t_p, brace_angle, pr_col["d"], pr_bm_bot["d"], pr["OD"]
        )
        hinge_dist_bot[level_counter] = res[1]
        gusset_avg_buckl_len_bot[level_counter] = res[2]
        pl_a_bot[level_counter] = trial_pl_a
        pl_b = trial_pl_a * np.tan(brace_angle)
        pl_b_bot[level_counter] = pl_b

        trial_pl_a = np.floor(l_c * 4.00)
        pl_a = fsolve(
            optim_gusset_plate_geometry,
            trial_pl_a,
            args=(l_c, t_p, brace_angle, pr_col["d"], pr_bm_top["d"], pr["OD"]),
        )
        assert len(pl_a) == 1
        pl_a_val = pl_a[0]
        # round up to the nearest 1/16 of an inch
        pl_a_val = np.ceil(pl_a_val * 16.00) / 16.00
        res = calc_gusset_plate_geometry(
            pl_a_val, t_p, brace_angle, pr_col["d"], pr_bm_top["d"], pr["OD"]
        )
        hinge_dist_top[level_counter] = res[1]
        gusset_avg_buckl_len_top[level_counter] = res[2]
        pl_a_top[level_counter] = trial_pl_a
        pl_b = trial_pl_a * np.tan(brace_angle)
        pl_b_top[level_counter] = pl_b

        buckling_length = (
            np.sqrt(
                (
                    hi_diff[level_counter - 1]
                    - pr_bm_bot["d"] / 2.00
                    + pr_bm_top["d"] / 2.00
                )
                ** 2
                + (25.00 * 12.00) ** 2
            )
            - hinge_dist_top[level_counter]
            - hinge_dist_bot[level_counter]
            + 8.00 * t_p
        )
        brace_buckling_length[level_counter] = buckling_length
        # brace compresive strength
        f_e = np.pi**2 * steel_e / (buckling_length / pr["rx"]) ** 2  # ksi
        f_cre = 0.658 ** (r_y_br * (r_y_br * f_y_br) / f_e) * (
            r_y_br * f_y_br
        )  # ksi
        p_nc = min(
            (1.0 / 0.877) * f_cre * pr["A"], r_y_br * f_y_br * pr["A"]
        )  # kips (obviously the first will control..)
        # checking if that is the case
        assert r_y_br * f_y_br > (1.0 / 0.877) * f_cre
        p_ncs[level_counter] = p_nc
        p_uts[level_counter] = p_ut

    # note:
    # print values of pl_a_top and _bot and check if they make sense

    # brace capacity check
    brace_capacity_ratios: dict[int, float] = {}
    for level_counter in range(1, num_lvls + 1):
        p_ucs = []
        for combo in design_combos:
            # retrieve brace: we have them in a list!
            comp = model_braces[level_counter - 1]
            elm = list(comp.elements.values())[0]
            # get demand
            p_ucs.append(
                -combo.envelope_basic_forces(elm, 2)[0].iloc[0, 0] / 1000.00
            )
        p_uc = max(p_ucs)
        # print(level_counter, p_uc)
        brace_capacity_ratios[level_counter] = p_uc / (0.90 * p_ncs[level_counter])
    print("Brace Strength Check Capacity Ratios")
    print([f"{brace_capacity_ratios[x]:.2f}" for x in range(1, num_lvls + 1)])
    print()
    for lvl_idx in range(1, num_lvls + 1):
        msgs += get_warning(
            brace_capacity_ratios[lvl_idx], "<", 1.00, "brace capacity"
        )

    print("Brace element slenderness")
    vals = [
        brace_buckling_length[level_tag]
        / brace_secs[f"level_{level_tag}"].properties["ry"]
        for level_tag in range(1, num_lvls + 1)
    ]
    print([f"{val:.2f}" for val in vals])
    print()
    for lvl_idx in range(num_lvls):
        msgs += get_warning(vals[lvl_idx], "<", 200.00, "brace slenderness")

    # beam and column strength check

    # For beam and column capacity check for fully developed brace
    # forces, we consider the fully developed strength of the braces
    # acting on the rest of the members, considering a swaying state
    # in both directions.  We use the model we stored earlier,
    # containing only the beams and columns. Inertial forces are not
    # accounted for in this analysis, and we can ignore P-Delta
    # effects for this capacity check, therefore we don't need the
    # leaning column.
    #
    # We only check for axial loads.
    # From "Ductile Design of Steel Structures", Bruneau et al. (2011):
    #     Two plastic mechanism analyses are performed on the
    #     frame. These are intended to capture both axial forces
    #     corresponding to brace inelastic action and flexural forces
    #     at the beams intersected by braces along their length.
    #     Although it is anticipated that these brace forces
    #     correspond to large drifts, and that columns may develop
    #     significant flexural forces at these drifts (due to fixity
    #     at beams or varying story drifts), these analyses are not
    #     intended to determine such flexural forces. Indeed, it is
    #     permitted to neglect them, under the assumption that limited
    #     flexural yielding in the column may be tolerated as long as
    #     overall buckling is precluded.
    #

    # add supports to the bareframe model
    query = ElmQuery(mdl_bareframe)
    for node in mdl_bareframe.list_of_all_nodes():
        node.restraint = [False, True, False, True, False, True]
    for node in mdl_bareframe.levels[0].nodes.values():
        node.restraint = [True] * 6
    # also support the nodes against movement in the X direction
    for lvl_idx in range(num_lvls):
        node = query.search_node_lvl(300.00, 0.00, lvl_idx + 1)
        node.restraint[0] = True

    # from osmg.graphics.preprocessing_3d import show
    # show(mdl_bareframe)

    # print('\n\n')
    for analysis_stage in ("with_braces", "buckling_stage", "post-buckling"):
        if analysis_stage == "with_braces":
            combos = design_combos
        else:
            # add loadcases and forces
            lc_bareframe_left = LoadCase("bareframe_left", mdl_bareframe)
            lc_bareframe_right = LoadCase("bareframe_right", mdl_bareframe)
            for level_counter in range(1, num_lvls + 1):
                # determine fully developed forces
                if analysis_stage == "buckling_stage":
                    p_t = p_uts[level_counter] * 1e3  # lb
                    p_c = p_ncs[level_counter] * 1e3  # lb
                else:
                    p_t = p_uts[level_counter] * 1e3  # lb
                    p_c = 0.30 * p_ncs[level_counter] * 1e3  # lb
                # print(level_counter, f'{p_t/1e3:.2f}', f'{p_c/1e3:.2f}')
                angle = np.arctan2((hi_diff[level_counter - 1]) / 12.00, 25.00)
                # forces coming from left brace
                nd_bot = query.search_node_lvl(0.00, 0.00, level_counter - 1)
                assert nd_bot
                nd_top = query.search_node_lvl(25.00 * 12.00, 0.00, level_counter)
                assert nd_top
                lc_bareframe_left.node_loads[nd_bot.uid].add(
                    np.array(
                        (
                            -p_c * np.cos(angle),
                            0.00,
                            -p_c * np.sin(angle),
                            0.00,
                            0.00,
                            0.00,
                        )
                    )
                )
                lc_bareframe_right.node_loads[nd_bot.uid].add(
                    np.array(
                        (
                            +p_t * np.cos(angle),
                            0.00,
                            +p_t * np.sin(angle),
                            0.00,
                            0.00,
                            0.00,
                        )
                    )
                )
                lc_bareframe_left.node_loads[nd_top.uid].add(
                    np.array(
                        (
                            +p_c * np.cos(angle),
                            0.00,
                            +p_c * np.sin(angle),
                            0.00,
                            0.00,
                            0.00,
                        )
                    )
                )
                lc_bareframe_right.node_loads[nd_top.uid].add(
                    np.array(
                        (
                            -p_t * np.cos(angle),
                            0.00,
                            -p_t * np.sin(angle),
                            0.00,
                            0.00,
                            0.00,
                        )
                    )
                )
                # forces coming from right brace
                nd_bot = query.search_node_lvl(
                    50.00 * 12.00, 0.00, level_counter - 1
                )
                assert nd_bot
                nd_top = query.search_node_lvl(25.00 * 12.00, 0.00, level_counter)
                assert nd_top
                lc_bareframe_left.node_loads[nd_bot.uid].add(
                    np.array(
                        (
                            -p_t * np.cos(angle),
                            0.00,
                            +p_t * np.sin(angle),
                            0.00,
                            0.00,
                            0.00,
                        )
                    )
                )
                lc_bareframe_right.node_loads[nd_bot.uid].add(
                    np.array(
                        (
                            +p_c * np.cos(angle),
                            0.00,
                            -p_c * np.sin(angle),
                            0.00,
                            0.00,
                            0.00,
                        )
                    )
                )
                lc_bareframe_left.node_loads[nd_top.uid].add(
                    np.array(
                        (
                            +p_t * np.cos(angle),
                            0.00,
                            -p_t * np.sin(angle),
                            0.00,
                            0.00,
                            0.00,
                        )
                    )
                )
                lc_bareframe_right.node_loads[nd_top.uid].add(
                    np.array(
                        (
                            -p_c * np.cos(angle),
                            0.00,
                            +p_c * np.sin(angle),
                            0.00,
                            0.00,
                            0.00,
                        )
                    )
                )

            static_anl_bareframe = solver.StaticAnalysis(
                mdl_bareframe,
                {
                    "bareframe_left": lc_bareframe_left,
                    "bareframe_right": lc_bareframe_right,
                },
            )

            static_anl_bareframe.run()

            # from osmg.graphics.postprocessing_3d import show_deformed_shape
            # from osmg.graphics.postprocessing_3d import show_basic_forces
            # show_deformed_shape(
            #     static_anl_bareframe, 'bareframe_left', 0, 0.00, True)
            # show_basic_forces(
            #     static_anl_bareframe, 'bareframe_left',
            #     0, 1.00, 0.00, 0.00, 0.00, 0.00, 10,
            #     force_conversion=1.00/1000.00,
            #     moment_conversion=1.00/12.00/1000.00,
            # )
            # show_deformed_shape(
            #     static_anl_bareframe, 'bareframe_right', 0, 0.00, True)
            # show_basic_forces(
            #     static_anl_bareframe, 'bareframe_right',
            #     0, 1.00, 0.00, 0.00, 0.00, 0.00, 10,
            #     force_conversion=1.00/1000.00,
            #     moment_conversion=1.00/12.00/1000.00,
            # )

            # create a combination

            bareframe_combo = LoadCombination(
                mdl_bareframe,
                {
                    "+": [(1.00, static_anl_bareframe, "bareframe_left")],
                    "-": [(1.00, static_anl_bareframe, "bareframe_right")],
                },
            )

            # from osmg.graphics.postprocessing_3d import show_basic_forces_combo
            # # units are in kip, kip-ft
            # show_basic_forces_combo(
            #     bareframe_combo, 1.00, 0.00, 0.00, 0.00, 0.00,
            #     30, 1.00/1000.00, 1.00/1000.00/12.00, True,
            # )

            combos = (bareframe_combo,)

        # for the defined combinations of that analysis case,
        # check for member adequacy
        element_capacity_ratios: list[float] = []
        for elm in mdl_bareframe.list_of_elements():
            if not isinstance(elm, ElasticBeamColumn):
                continue
            # ~~~ capacity ~~~
            # gather all elements, skipping panel zone parts
            if not isinstance(elm, ElasticBeamColumn):
                continue
            elm_len = elm.clear_length()
            if elm_len < 5.00 * 12.00:
                continue

            # determine what kind of element it is and at what story
            if elm.parent_component.component_purpose == "vertical_component":
                elm_type = "column"
            else:
                elm_type = "beam"
            n_uid = list(elm.parent_component.external_nodes.values())[0].uid
            elmlvl = 0
            for lvl in mdl.levels.values():
                if n_uid in lvl.nodes:
                    elmlvl = lvl.uid
            assert elmlvl != 0

            # section properties
            prop = elm.section.properties
            assert prop is not None
            klr_x = 1.00 * elm_len / prop["rx"]
            if elm_type == "column":
                klr_y = 1.00 * elm_len / prop["ry"]
            else:
                klr_y = 1.00 * (elm_len / 2.00) / prop["ry"]
            klr = max(klr_x, klr_y)
            fe = np.pi**2 * steel_e / klr**2
            if klr < 4.71 * np.sqrt(steel_e / f_y_fr):
                f_cr = 0.658 ** (f_y_fr / fe) * f_y_fr  # (AISC E3-2)
            else:
                f_cr = 0.877 * fe  # (AISC E3-3)
            pn = f_cr * prop["A"]

            # ~~~ demand ~~~
            p_ucs = []
            for combo in combos:
                # get demand
                f1 = pd.concat(
                    (
                        combo.envelope_basic_forces(elm, 2)[0],
                        combo.envelope_basic_forces(elm, 2)[1],
                    )
                )[
                    "nx"
                ].to_numpy()  # axial load, all cases
                f1 = f1[f1 < 0.00]  # only consider compressive load
                if len(f1) == 0:
                    continue
                p_uc = -min(f1) / 1000.00  # peak compressive load
                p_ucs.append(p_uc)
            if len(p_ucs) == 0:
                continue  # only tension for that member
            p_uc_max = max(p_ucs)
            element_capacity_ratios.append(p_uc_max / (0.90 * pn))
            msgs += get_warning(
                p_uc_max / (0.90 * pn),
                "<",
                1.00,
                f"frame capacity, {elm_type} lvl{elmlvl}",
            )

        print(f"Frame Strength Check Capacity Ratios: {analysis_stage}")
        print([f"{x:.2f}" for x in element_capacity_ratios])
        print()

    # calculate structural weight (lb)
    total_weight = 0.00
    for level_counter in range(1, num_lvls + 1):
        level_tag = "level_" + str(level_counter)
        bm_w = beam_secs[level_tag].properties["W"]
        col_w = col_secs[level_tag].properties["W"]
        br_w = brace_secs[level_tag].properties["W"]
        total_weight += bm_w * 25.00 * 2.00
        total_weight += col_w * hi_diff[level_counter - 1] / 12 * 3.00
        total_weight += (
            br_w * np.sqrt((hi_diff[level_counter - 1] / 12) ** 2 + 25.00**2)
        ) * 2.00

    print()
    print(f"  steel weight:  {total_weight:.2f} lb")
    print(
        "  steel weight/bays/levels:  " f"{total_weight/num_lvls/2.00/1000:.1f} kips"
    )
    print(f"  warnings: {msgs}")

    if show_metadata:
        print("~~~ Showing design metadata ~~~\n")
        print("brace_buckling_length")
        print([f"{ln:.4f}" for ln in brace_buckling_length.values()])
        print()
        print("brace_l_c")
        print([f"{x:.4f}" for x in brace_l_c.values()])
        print()
        print("gusset_t_p")
        print([f"{x:.4f}" for x in gusset_t_p.values()])
        print()
        print("gusset_avg_buckl_len")
        print([f"{x:.4f}" for x in gusset_avg_buckl_len_top.values()])
        # print([f'{x:.4f}' for x in gusset_avg_buckl_len_bot.values()])
        print()
        print("hinge_dist")
        print([f"{x:.4f}" for x in hinge_dist_top.values()])
        # print([f'{x:.4f}' for x in hinge_dist_bot.values()])
        print()
        print("plate_a")
        print([f"{x:.4f}" for x in pl_a_top.values()])
        # print([f'{x:.4f}' for x in hinge_dist_bot.values()])
        print()
        print("plate_b")
        print([f"{x:.4f}" for x in pl_b_top.values()])
        # print([f'{x:.4f}' for x in hinge_dist_bot.values()])
        print()
