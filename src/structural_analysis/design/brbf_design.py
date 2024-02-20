"""
Design a n-story BRBF system
"""

import copy
from osmg.model import Model
from osmg.gen.component_gen import TrussBarGenerator
from osmg.gen.component_gen import BeamColumnGenerator
from osmg.gen.section_gen import SectionGenerator
from osmg.gen.query import ElmQuery
from osmg import defaults
from osmg.ops.element import ElasticBeamColumn
from osmg.ops.element import TrussBar
from osmg.ops.uniaxial_material import Elastic
from osmg.gen.zerolength_gen import release_6
from osmg.load_case import LoadCase
from osmg import solver
from osmg.ops.section import ElasticSection
from osmg import common
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass
from osmg.postprocessing.design import LoadCombination
from osmg.gen.mesh_shapes import rect_mesh

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp

nparr = npt.NDArray[np.float64]

# pylint: disable=invalid-name


def design_brbf_lrrs(
    num_lvls,
    beams: dict[int, list[str]],
    columns: dict[int, list[str]],
    coeff: list[int],
    brace_core_areas: dict[int, float],
    beam_udls_dead: dict[str, float],
    beam_udls_live: dict[str, float],
    lvl_weight: dict[str, float],
    design_params: dict[str, float],
    site_characteristics: dict[str, float],
    tmax_params: dict[str, float],
    mlp_periods: nparr,
    mlp_des_spc: nparr,
    num_braces: int,
):
    """
    Design a n-story BRBF system

    Args:
      num_braces: How many lines of braced bays exist in the plan
      (not # of individual braces)

    """

    # The stiffness modification factor as a function of core area and
    # workpoint length is highly nonlinear and depends on the
    # manufacturer. For this study we use the design aid provided by
    # CoreBrace (https://corebrace.com/resources/) for bolted BRBs.

    # load the stiffness modification factor (smf) data
    df_smf = pd.read_csv(
        "src/structural_analysis/design/brbf_stiffness_modification_factors.csv",
        skiprows=6,
        index_col=(0),
    )
    df_smf = (
        df_smf.assign(WorkPtLen_ft=np.sqrt(df_smf.Bay_ft**2 + df_smf.Height_ft**2))
        .drop(columns=["Bay_ft", "Height_ft"])
        .reset_index()
        .set_index(["Asc_in2", "WorkPtLen_ft"])
    )

    # cast to numpy arrays
    points_smf = np.array(df_smf.index.to_list()).tolist()
    values_smf = df_smf.to_numpy().reshape(-1)
    # generate interpolation function
    interp_smf = sp.interpolate.LinearNDInterpolator(points_smf, values_smf)

    # # test
    # trial_point = np.array((4.0, 25.0))
    # value = interp_smf(trial_point)[0]

    # import matplotlib.pyplot as plt
    # delta = 0.1
    # x = np.arange(2.0, 30.0, delta)
    # y = np.arange(20.6, 40.0, delta)
    # X, Y = np.meshgrid(x, y)
    # Z = interp_smf(X, Y)
    # z_min, z_max = Z.min(), Z.max()
    # fig, ax = plt.subplots()
    # c = ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=z_min, vmax=z_max)
    # ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    # fig.colorbar(c, ax=ax)
    # plt.xlabel('Core Area [in$^2$]')
    # plt.ylabel('Workpoint Length [ft]')
    # plt.title('CoreBrace Stiffness Modification Factor')
    # plt.show()
    # # It's kind of strange how non-smooth this is.

    df_acs = pd.read_csv(
        "src/structural_analysis/design/brbf_approximate_casing_sizes.csv",
        skiprows=6,
        index_col=(0),
    )
    df_acs = (
        df_acs.assign(WorkPtLen_ft=np.sqrt(df_acs.Bay_ft**2 + df_acs.Height_ft**2))
        .drop(columns=["Bay_ft", "Height_ft"])
        .reset_index()
        .set_index(["Asc_in2", "WorkPtLen_ft"])
    )

    # cast to numpy arrays
    points_acs = np.array(df_acs.index.to_list()).tolist()
    values_acs = np.array(
        [
            float(x.replace("t", "").replace("p", ""))
            for x in df_acs.to_numpy().reshape(-1)
        ]
    )
    # yes, we will draw everything as squares for our visualization purposes..
    # generate interpolation function
    interp_acs = sp.interpolate.LinearNDInterpolator(points_acs, values_acs)

    # Stage: Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # BRB design parameters
    beta_factor = 1.20  # compression overstrength adjustment
    omega_factor = 1.60  # strain hardening adjustment
    f_y_sc = 38.00  # steel core nominal yield strength [ksi]
    r_y_sc = 1.20  # overstrength factor
    rough_density = 150.00 / (12.00) ** 3  # lb/in3, approximate brb weight

    # Stage: Model definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    beam_coeff_lvl: dict[int, int] = {}
    col_coeff_lvl: dict[int, int] = {}
    running_idx = 0
    for lvl_idx in range(1, num_lvls + 1):
        beam_coeff_lvl[lvl_idx] = running_idx
        running_idx += 1
    for lvl_idx in range(1, num_lvls + 1):
        col_coeff_lvl[lvl_idx] = running_idx
        running_idx += 1

    mdl = Model("design_model")
    mdl.settings.imperial_units = True
    trg = TrussBarGenerator(mdl)
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
    for lvl_idx in range(1, num_lvls + 1):
        beam_secs[f"level_{lvl_idx}"] = section_from_index(
            coeff[beam_coeff_lvl[lvl_idx]], beams[lvl_idx], "W"
        )
        col_secs[f"level_{lvl_idx}"] = section_from_index(
            coeff[col_coeff_lvl[lvl_idx]], columns[lvl_idx], "W"
        )

    print("Sections")
    print("Beams")
    print([sec.name for sec in beam_secs.values()])
    print("Columns")
    print([sec.name for sec in col_secs.values()])
    print("BRBs")
    print([f"{area:.2f}" for area in brace_core_areas.values()])
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
        area = brace_core_areas[level_counter + 1]  # in2
        assert 2.00 <= area <= 30.00
        # interpolate CoreBrace catalog for the stiffness modification factor
        # area in in2, bay width and story height in ft.
        workpoint_length = np.sqrt(
            (25.00 * 12.00) ** 2 + (hi_diff[level_counter]) ** 2
        )  # in
        trial_point = np.array((area, workpoint_length / 12.00))
        stiffness_mod_factor = interp_smf(trial_point)[0]
        casing_size = interp_acs(trial_point)[0]
        assert not np.isnan(stiffness_mod_factor)
        # calculate the effective stiffness
        brb_e_eff = 29000000.00 * stiffness_mod_factor  # [lb/in]
        # create linear material
        mat = Elastic(
            uid=mdl.uid_generator.new("uniaxial material"),
            name=f"BRB_{level_counter+1}_left",
            e_mod=brb_e_eff,
        )
        # create outside shape
        outside_shape = rect_mesh(casing_size, casing_size)
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])
        if level_counter % 2 == 0:
            pt_A = np.array((0.00, 0.00))
            pt_B = np.array((25.00 * 12.00, 0.00))
        else:
            pt_A = np.array((25.00 * 12.00, 0.00))
            pt_B = np.array((0.00, 0.00))

        added_brace = trg.add(
            pt_B[0],
            pt_B[1],
            level_counter + 1,
            np.array((0.00, 0.00, vertical_offsets[level_counter + 1])),
            # 'middle_front',
            "centroid",
            pt_A[0],
            pt_A[1],
            level_counter,
            np.array((0.00, 0.00, vertical_offsets[level_counter])),
            # 'middle_back',
            "centroid",
            "Corotational",
            area=area,
            mat=mat,
            outside_shape=outside_shape,
            weight_per_length=rough_density * casing_size**2,  # lb/in
        )
        model_braces.append(added_brace)

    # right braces
    for level_counter in range(num_lvls):
        area = brace_core_areas[level_counter + 1]  # in2
        assert 2.00 <= area <= 30.00
        # interpolate CoreBrace catalog for the stiffness modification factor
        # area in in2, bay width and story height in ft.
        workpoint_length = np.sqrt(
            (25.00 * 12.00) ** 2 + (hi_diff[level_counter]) ** 2
        )  # in
        trial_point = np.array((area, workpoint_length / 12.00))
        stiffness_mod_factor = interp_smf(trial_point)[0]
        casing_size = interp_acs(trial_point)[0]
        assert not np.isnan(stiffness_mod_factor)
        # calculate the effective stiffness
        brb_e_eff = 29000000.00 * stiffness_mod_factor  # [lb/in]
        # create linear material
        mat = Elastic(
            uid=mdl.uid_generator.new("uniaxial material"),
            name=f"BRB_{level_counter+1}_right",
            e_mod=brb_e_eff,
        )
        # create outside shape
        outside_shape = rect_mesh(casing_size, casing_size)
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])
        if level_counter % 2 == 0:
            pt_A = np.array((50.00 * 12.00, 0.00))
            pt_B = np.array((25.00 * 12.00, 0.00))
        else:
            pt_A = np.array((25.00 * 12.00, 0.00))
            pt_B = np.array((50.00 * 12.00, 0.00))

        added_brace = trg.add(
            pt_B[0],
            pt_B[1],
            level_counter + 1,
            np.array((0.00, 0.00, vertical_offsets[level_counter + 1])),
            # 'middle_back',
            "centroid",
            pt_A[0],
            pt_A[1],
            level_counter,
            np.array((0.00, 0.00, vertical_offsets[level_counter])),
            # 'middle_front',
            "centroid",
            "Corotational",
            area=area,
            mat=mat,
            outside_shape=outside_shape,
            weight_per_length=rough_density * casing_size**2,
        )

    # leaning column
    for level_counter in range(num_lvls):
        level_tag = "level_" + str(level_counter + 1)
        mdl.levels.set_active([level_counter + 1])
        pt = np.array((75.00 * 12.00, 0.00))

        sec = beam_secs[level_tag]
        d = sec.properties["d"]

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
        pt = np.array((-25.00 * 12.00, 0.00))

        sec = beam_secs[level_tag]
        d = sec.properties["d"]

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

    p_nodes_i = []
    p_nodes_j = []
    for i in range(1, num_lvls + 1):
        p_nodes_i.append(query.search_node_lvl(-25.00 * 12.00, 0.00, i))
        p_nodes_j.append(query.search_node_lvl(75.00 * 12.00, 0.00, i))

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
    # show(mdl, extrude=True)
    # show(subset_model, extrude=True)

    # assign loads

    lc_dead = LoadCase("dead", mdl)
    self_mass(mdl, lc_dead)
    self_weight(mdl, lc_dead)
    lc_live = LoadCase("live", mdl)

    for level_counter in range(1, num_lvls + 1):
        level_tag = "level_" + str(level_counter)
        for xpt in [12.5 * 12.00, 37.50 * 12.00]:
            comp = query.retrieve_component(xpt, 0.00, level_counter)
            assert comp
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
    # show(subset_model, lc_dead, extrude=False)
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
            f = sp.interpolate.interp1d(x, y)
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
                f = sp.interpolate.interp1d(x, y)
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
    #     modal_analysis, 'modal', 2, 0.00,
    #     True, subset_model=subset_model
    # )
    # show_basic_forces(
    #     modal_analysis, 'modal', 0, 2.00, 0.00, 0.00,
    #     0.00, 0.00, 10, 1.00, 1.00,
    #     global_axes=True, subset_model=subset_model)

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

    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # show_deformed_shape(
    #     elf_anl, 'elf', 0, 0.00, False, subset_model=subset_model)

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
    #     subset_model=subset_model
    # )

    # design combinations

    static_anl = solver.StaticAnalysis(mdl, {"dead": lc_dead, "live": lc_live})
    static_anl.run()

    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # from osmg.graphics.postprocessing_3d import show_basic_forces
    # show_deformed_shape(
    #     static_anl, 'live', 0, 0.00, True, subset_model=subset_model)
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
    assert rsa.vb_modal is not None
    ts = rsa.anl.results["modal"].periods

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

    # Stage: Global checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # construct and issue warnings
    msgs = "\n"

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

    # brace capacity check

    brace_capacity_ratios: dict[int, float] = {}
    for level_counter in range(1, num_lvls + 1):
        level_tag = "level_" + str(level_counter)
        p_ucs = []
        for combo in design_combos:
            # retrieve brace: we have them in a list!
            comp = model_braces[level_counter - 1]
            elm = list(comp.elements.values())[2]
            assert isinstance(elm, TrussBar)
            # get demand
            p_ucs.append(
                -combo.envelope_basic_forces(elm, 2)[0].iloc[0, 0] / 1000.00
            )
        p_uc = max(p_ucs)  # this is kips
        area = brace_core_areas[level_counter]  # in2
        brace_capacity_ratios[level_counter] = p_uc / (0.90 * f_y_sc * area)
        msgs += get_warning(
            p_uc / (0.90 * f_y_sc * area), "<", 1.00, "brace capacity"
        )

    print("Brace Strength Check Capacity Ratios")
    print([f"{brace_capacity_ratios[lc]:.2f}" for lc in range(1, num_lvls + 1)])
    print()

    # beam and column strength check

    p_ucs = []
    element_capacity_ratios: list[float] = []
    for elm in mdl_bareframe.list_of_elements():
        # gather all elements, skipping panel zone parts
        if not isinstance(elm, ElasticBeamColumn):
            continue
        elm_len = elm.clear_length()
        if elm_len < 5.00 * 12.00:
            continue
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
        prop = elm.section.properties
        assert prop is not None
        klr_x = 1.00 * elm_len / prop["rx"]
        if elm_type == "column":
            klr_y = 1.00 * elm_len / prop["ry"]
        else:
            klr_y = 1.00 * (elm_len / 2.00) / prop["ry"]
        klr = max(klr_x, klr_y)
        fe = np.pi**2 * 29000.00 / klr**2
        if klr < 4.71 * np.sqrt(29000.00 / 50.00):
            f_cr = 0.658 ** (50.00 / fe) * 50.00  # (AISC E3-2)
        else:
            f_cr = 0.877 * fe  # (AISC E3-3)
        pn = f_cr * prop["A"]
        p_ucs = []
        for combo in design_combos:
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
                p_uc = 0.00
            else:
                p_uc = -min(f1) / 1000.00  # peak compressive load
            p_ucs.append(p_uc)
        p_uc_max = max(p_ucs)
        element_capacity_ratios.append(p_uc_max / (0.90 * pn))
        msgs += get_warning(
            p_uc_max / (0.90 * pn),
            "<",
            1.00,
            f"frame capacity, {elm_type} lvl{elmlvl}",
        )

    # beam and column capacity check for fully developed brace forces

    # for this, we consider the fully developed strength of the braces
    # acting on the rest of the members, considering a swaying state
    # in both directions.
    # We use the model we stored earlier, containing only the beams and
    # columns. Inertial forces are not accounted for in this analysis,
    # and we can ignore P-Delta effects for this capacity check, therefore
    # we don't need the leaning column.

    # from osmg.graphics.preprocessing_3d import show
    # show(mdl_bareframe)

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

    # add loadcases and forces
    lc_bareframe_left = LoadCase("bareframe_left", mdl_bareframe)
    lc_bareframe_right = LoadCase("bareframe_right", mdl_bareframe)
    for level_counter in range(1, num_lvls + 1):
        # determine fully developed forces
        area = brace_core_areas[level_counter]  # in2
        p_t = area * f_y_sc * r_y_sc * omega_factor * 1e3  # lb
        p_c = area * f_y_sc * r_y_sc * omega_factor * beta_factor * 1e3  # lb
        angle = np.arctan2((hi_diff[level_counter - 1]) / 12.00, 25.00)
        # forces coming from left brace
        nd_bot = query.search_node_lvl(0.00, 0.00, level_counter - 1)
        assert nd_bot
        nd_top = query.search_node_lvl(25.00 * 12.00, 0.00, level_counter)
        assert nd_top
        lc_bareframe_left.node_loads[nd_bot.uid].add(
            np.array(
                (-p_c * np.cos(angle), 0.00, -p_c * np.sin(angle), 0.00, 0.00, 0.00)
            )
        )
        lc_bareframe_right.node_loads[nd_bot.uid].add(
            np.array(
                (+p_t * np.cos(angle), 0.00, +p_t * np.sin(angle), 0.00, 0.00, 0.00)
            )
        )
        lc_bareframe_left.node_loads[nd_top.uid].add(
            np.array(
                (+p_c * np.cos(angle), 0.00, +p_c * np.sin(angle), 0.00, 0.00, 0.00)
            )
        )
        lc_bareframe_right.node_loads[nd_top.uid].add(
            np.array(
                (-p_t * np.cos(angle), 0.00, -p_t * np.sin(angle), 0.00, 0.00, 0.00)
            )
        )
        # forces coming from right brace
        nd_bot = query.search_node_lvl(50.00 * 12.00, 0.00, level_counter - 1)
        assert nd_bot
        nd_top = query.search_node_lvl(25.00 * 12.00, 0.00, level_counter)
        assert nd_top
        lc_bareframe_left.node_loads[nd_bot.uid].add(
            np.array(
                (-p_t * np.cos(angle), 0.00, +p_t * np.sin(angle), 0.00, 0.00, 0.00)
            )
        )
        lc_bareframe_right.node_loads[nd_bot.uid].add(
            np.array(
                (+p_c * np.cos(angle), 0.00, -p_c * np.sin(angle), 0.00, 0.00, 0.00)
            )
        )
        lc_bareframe_left.node_loads[nd_top.uid].add(
            np.array(
                (+p_t * np.cos(angle), 0.00, -p_t * np.sin(angle), 0.00, 0.00, 0.00)
            )
        )
        lc_bareframe_right.node_loads[nd_top.uid].add(
            np.array(
                (-p_c * np.cos(angle), 0.00, +p_c * np.sin(angle), 0.00, 0.00, 0.00)
            )
        )

    static_anl_bareframe = solver.StaticAnalysis(
        mdl_bareframe,
        {"bareframe_left": lc_bareframe_left, "bareframe_right": lc_bareframe_right},
    )

    static_anl_bareframe.settings.restrict_dof = [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    static_anl_bareframe.run()

    # from osmg.graphics.postprocessing_3d import show_deformed_shape
    # show_deformed_shape(
    #     static_anl_bareframe, 'bareframe_left', 0, 0.00, True)
    # from osmg.graphics.postprocessing_3d import show_basic_forces
    # show_basic_forces(
    #     static_anl_bareframe, 'bareframe_left',
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
    # show_basic_forces_combo(
    #     bareframe_combo, 1.00, 0.00, 0.00, 0.00, 0.00,
    #     30, 1.00/1000.00, 1.00/1000.00/12.00, True,
    # )
    # units are in kip, kip-ft

    # retrieve basic forces and check member capacity

    # We only check for axial loads.
    # From "Ductile Design of Steel Structures", Bruneau et al. (2011):
    #     This capacity-design procedure for determining required
    #     strengths of frame members is intended to prevent beam and
    #     column buckling. It is not intended to prevent limited yielding
    #     in the frame members. Accordingly, moments in these members are
    #     not considered in conjunction with these large axial
    #     forces. Rather, sufficient rotational capacity is ensured
    #     through the use of highly compact sections and the use of either
    #     fully restrained beam to column connections capable of resisting
    #     moments corresponding to the flexural strength of the beam (or
    #     column) or the use of "simple" connections capable of
    #     accommodating a rotation of 0.025 radians.
    #
    p_ucs = []
    combo = bareframe_combo
    element_capacity_ratios: list[float] = []
    for elm in mdl_bareframe.list_of_elements():
        # gather all elements, skipping panel zone parts
        if not isinstance(elm, ElasticBeamColumn):
            continue
        elm_len = elm.clear_length()
        if elm_len < 5.00 * 12.00:
            continue
        if elm.parent_component.component_purpose == "vertical_component":
            elm_type = "column"
        else:
            elm_type = "beam"
        n_uid = list(elm.parent_component.external_nodes.values())[0].uid
        elmlvl = 0
        for lvl in mdl_bareframe.levels.values():
            if n_uid in lvl.nodes:
                elmlvl = lvl.uid
        assert elmlvl != 0
        prop = elm.section.properties
        assert prop is not None
        klr_x = 1.00 * elm_len / prop["rx"]
        klr_y = 1.00 * elm_len / prop["ry"]
        klr = max(klr_x, klr_y)
        fe = np.pi**2 * 29000.00 / klr**2
        if klr < 4.71 * np.sqrt(29000.00 / 50.00):
            f_cr = 0.658 ** (50.00 / fe) * 50.00  # (AISC E3-2)
        else:
            f_cr = 0.877 * fe  # (AISC E3-3)
        pn = f_cr * prop["A"]
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
        element_capacity_ratios.append(p_uc / (0.90 * pn))
        msgs += get_warning(
            p_uc / (0.90 * pn),
            "<",
            1.00,
            "frame capacity (fully developed brace forces), "
            f"{elm_type} lvl{elmlvl}",
        )

    print("Frame Strength Check Capacity Ratios")
    print([f"{x:.2f}" for x in element_capacity_ratios])
    print()

    print(f"  warnings: {msgs}")
