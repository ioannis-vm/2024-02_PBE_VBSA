"""
Perform VBSA using pelicun
"""

import os
import argparse
import numpy as np
import pandas as pd
from pelicun.assessment import Assessment
from pelicun import uq
from pelicun import base
from scipy.interpolate import interp1d
from src.util import read_study_param
from src.util import store_info

# pylint: disable=protected-access


parser = argparse.ArgumentParser()
parser.add_argument("--archetype")
parser.add_argument("--occupancy")
parser.add_argument("--hz")
parser.add_argument("--modeling_uncertainty_case")
parser.add_argument("--rv_group")
parser.add_argument("--input_dir_name")
parser.add_argument("--output_dir_name")
parser.add_argument("--out_prefix", default="vbsa")
parser.add_argument("--b_repl_factor", default=1.00)

args = parser.parse_args()
archetype = args.archetype
occupancy = args.occupancy
hz = args.hz
modeling_uncertainty = args.modeling_uncertainty_case
input_dir_name = args.input_dir_name
output_dir_name = args.output_dir_name
out_prefix = args.out_prefix
b_repl_factor = float(args.b_repl_factor)

# archetype = 'smrf_3_ii'
# occupancy = 'healthcare'
# hz = '1'
# modeling_uncertainty = 'low'
# input_dir_name = 'edp'
# output_dir_name = 'test'
# out_prefix = 'vbsa'
# b_repl_factor = 1.00

num_realizations = 1000
num_repetitions = (0, 10)
loss_types = ("Cost", "Time")
total_loss_thresholds = (1.0, 0.4)

# split the output to directories to avoid having a huge number of
# files in a single directory
output_dir_name = os.path.join(
    output_dir_name, archetype, occupancy, hz, modeling_uncertainty
)


modeling_uncertainty_val = {"low": 0.141, "medium": 0.353}

rv_filtering = {
    "EDP": {"include": ("EDP", "RID"), "exclude": ("excessiveRID",)},
    "EDP-PID": {"include": ("EDP-PID",), "exclude": tuple()},
    "EDP-PFV": {"include": ("EDP-PFV",), "exclude": tuple()},
    "EDP-PFA": {"include": ("EDP-PFA",), "exclude": tuple()},
    "EDP-RID": {"include": ("RID",), "exclude": ("excessiveRID",)},
    "CMP": {
        "include": ("CMP",),
        "exclude": ("collapse", "excessiveRID", "irreparable"),
    },
    "C-DS": {
        "include": ("FRG", "LSDS"),
        "exclude": ("collapse", "excessiveRID", "irreparable"),
    },
    "C-DS-FRG": {
        "include": ("FRG",),
        "exclude": ("collapse", "excessiveRID", "irreparable"),
    },
    "C-DS-LS": {
        "include": ("LSDS",),
        "exclude": ("collapse", "excessiveRID", "irreparable"),
    },
    "B-DS": {
        "include": (
            "FRG-collapse",
            "FRG-excessive",
            "FRG-irreparable",
            "LSDS-collapse",
            "LSDS-excessive",
            "LSDS-irreparable",
        ),
        "exclude": tuple(),
    },
    "B-DSc": {
        "include": (
            "FRG-collapse",
            "LSDS-collapse",
        ),
        "exclude": tuple(),
    },
    "B-DSe": {
        "include": (
            "FRG-excessive",
            "LSDS-excessive",
        ),
        "exclude": tuple(),
    },
    "C-DV": {"include": ("Cost", "Time"), "exclude": ("replacement",)},
    "B-DV": {
        "include": ("Cost-replacement", "Time-replacement"),
        "exclude": tuple(),
    },
}


rv_groups = rv_filtering.keys()


def rm_unnamed(string):
    """
    Fix column names after import
    """
    if "Unnamed: " in string:
        return ""
    return string


def matches(rv_name, rv_name_filter_tags, rv_name_filter_tag_exclude):
    """
    Checks if rv_name matches any of the tags in
    rv_name_filter_tags. If rv_name contains any tag from
    rv_name_filter_tag_exclude, it returns False. If rv_name contains
    any tag from rv_name_filter_tags, it returns True. If rv_name
    doesn't contain any filter tags, it returns False.

    Parameters:
    rv_name (str): The name to be checked against the filter tags.

    Returns:
    bool: True if rv_name contains a filter tag, False otherwise.
    """
    for etg in rv_name_filter_tag_exclude:
        if etg in rv_name:
            return False
    for tg in rv_name_filter_tags:
        if tg in rv_name:
            return True
    return False


# ---------------------------------- #
# Initialization                     #
# ---------------------------------- #

system, num_stories, risk_category = archetype.split("_")
num_stories = int(num_stories)


def main(repetition_counter):
    """
    Generate loss estimates for assessments A, B, C, and D, with all
    required rv_group assignments.
    """

    def demand_initialize_assessments():
        """
        initialize four Pelicun Assessment objects
        """
        asmts = {
            "A": Assessment({"PrintLog": False}),
            "B": Assessment({"PrintLog": False}),
        }
        for rv_group in rv_groups:
            asmts["C/" + rv_group] = Assessment({"PrintLog": False})
            asmts["D/" + rv_group] = Assessment({"PrintLog": False})
        for asmt in asmts.values():
            asmt.stories = num_stories
        return asmts

    asmts = demand_initialize_assessments()

    # ---------------------------------- #
    # Building Response Realizations     #
    # ---------------------------------- #

    def process_raw_demands():
        """
        load raw demands from building response output
        """
        path = f"results/response/{archetype}/{input_dir_name}/{hz}/response.parquet"
        raw_demands = pd.read_parquet(path)
        # remove collapse cases
        xxx = raw_demands["PID"].max(axis=1)
        idx = xxx[xxx < 0.06 + 1e-4].index  # we consider PID > 6% a collapse
        raw_demands = raw_demands.loc[idx, :]
        raw_demands.columns.names = ["type", "loc", "dir"]
        raw_demands.drop(["PVb"], axis=1, inplace=True)
        # add units
        units = []
        for col in raw_demands.columns:
            if col[0] == "PFA":
                units.append(["g"])
            elif col[0] == "PFV":
                units.append(["inps2"])
            elif col[0] == "PID":
                units.append(["rad"])
            else:
                raise ValueError(f"Invalid EDP type: {col[0]}")
        units_df = pd.DataFrame(dict(zip(raw_demands.columns, units)))
        units_df.index = ["Units"]
        raw_demands = pd.concat((units_df, raw_demands))
        return raw_demands

    raw_demands = process_raw_demands()

    def demand_sample_demand_distribution():
        """
        sample the demand distribution
        """
        for key in ("A", "B"):
            asmt = asmts[key]
            asmt.demand.load_sample(raw_demands)
            asmt.demand.calibrate_model(
                {
                    "ALL": {
                        "DistributionFamily": "lognormal",
                        "AddUncertainty": (
                            modeling_uncertainty_val[modeling_uncertainty]
                        ),
                    },
                    "PID": {
                        "DistributionFamily": "lognormal",
                        "TruncateLower": "",
                        "TruncateUpper": "0.10",
                        "AddUncertainty": (
                            modeling_uncertainty_val[modeling_uncertainty]
                        ),
                    },
                }
            )
            # asmt.demand.generate_sample({"SampleSize": num_realizations})
            # ... breaking it down to lower-level methods
            config = {"SampleSize": num_realizations}
            if asmt.demand.marginal_params is None:
                raise ValueError(
                    "Model parameters have not been specified. Either"
                    "load parameters from a file or calibrate the "
                    "model using raw demand data."
                )
            # Assessments A and B sample RVs
            asmt.demand._create_RVs(
                preserve_order=config.get("PreserveRawOrder", False)
            )
            asmt.demand._RVs.generate_sample(
                sample_size=num_realizations,
                method=asmt.demand._asmnt.options.sampling_method,
            )
            # replace the potentially existing raw sample with the generated one
            asmt.demand._sample = None

    demand_sample_demand_distribution()

    # Assessments C and D use the sampled RVs of assessments A and
    # B, with appropriate assignments depending on the RV group we
    # want to determine the sobol indices.

    def demand_initialize_rv_registry():
        """
        initialize RV registry
        """
        for rv_group in rv_groups:
            asmts["C/" + rv_group].demand._RVs = uq.RandomVariableRegistry(
                asmts["C/" + rv_group].options.rng
            )
            asmts["D/" + rv_group].demand._RVs = uq.RandomVariableRegistry(
                asmts["D/" + rv_group].options.rng
            )

    demand_initialize_rv_registry()

    def demand_add_demand_rvs():
        """
        add the RVs from assessments A and B
        """
        rv_names = asmts["A"].demand._RVs.RV.keys()
        assert rv_names == asmts["B"].demand._RVs.RV.keys()
        for rv_group in rv_groups:
            rv_name_filter_tags = rv_filtering[rv_group]["include"]
            rv_name_filter_tag_exclude = rv_filtering[rv_group]["exclude"]
            assert isinstance(rv_name_filter_tags, tuple)
            assert isinstance(rv_name_filter_tag_exclude, tuple)
            for rv_name in rv_names:
                # if it matches from A, it goes to C,
                # otherwise it goes to D.
                if matches(rv_name, rv_name_filter_tags, rv_name_filter_tag_exclude):
                    asmts["C/" + rv_group].demand._RVs.add_RV(
                        asmts["A"].demand._RVs.RV[rv_name]
                    )
                    asmts["D/" + rv_group].demand._RVs.add_RV(
                        asmts["B"].demand._RVs.RV[rv_name]
                    )
                else:
                    asmts["C/" + rv_group].demand._RVs.add_RV(
                        asmts["B"].demand._RVs.RV[rv_name]
                    )
                    asmts["D/" + rv_group].demand._RVs.add_RV(
                        asmts["A"].demand._RVs.RV[rv_name]
                    )
            # clear out the sample and add the units
            asmts["C/" + rv_group].demand._sample = None
            asmts["D/" + rv_group].demand._sample = None
            asmts["C/" + rv_group].demand.units = asmts["A"].demand.units
            asmts["D/" + rv_group].demand.units = asmts["B"].demand.units

    demand_add_demand_rvs()

    def demand_add_rid_and_sa():
        """
        add residual drift and Sa(T1)
        """
        demand_sample = {}
        demand_sample["A"] = asmts["A"].demand.save_sample()
        demand_sample["B"] = asmts["B"].demand.save_sample()
        for rv_group in rv_groups:
            demand_sample["C/" + rv_group] = asmts[
                "C/" + rv_group
            ].demand.save_sample()
            demand_sample["D/" + rv_group] = asmts[
                "D/" + rv_group
            ].demand.save_sample()

        # add residual drift based on FEMA P-58 simplified analysis equations
        delta_y = float(read_study_param(f"data/{archetype}/yield_dr")) / 100.00

        PID_A = demand_sample["A"]["PID"]
        PID_B = demand_sample["B"]["PID"]

        RID_A = asmts["A"].demand.estimate_RID(PID_A, {"yield_drift": delta_y})
        RID_B = asmts["B"].demand.estimate_RID(PID_B, {"yield_drift": delta_y})

        demand_sample_ext = {}
        demand_sample_ext["A"] = pd.concat([demand_sample["A"], RID_A], axis=1)
        demand_sample_ext["B"] = pd.concat([demand_sample["B"], RID_B], axis=1)

        for rv_group in rv_groups:
            rv_name_filter_tags = rv_filtering[rv_group]["include"]
            rv_name_filter_tag_exclude = rv_filtering[rv_group]["exclude"]
            assert isinstance(rv_name_filter_tags, tuple)
            assert isinstance(rv_name_filter_tag_exclude, tuple)

            if "RID" in rv_name_filter_tags:
                demand_sample_ext["C/" + rv_group] = pd.concat(
                    [demand_sample["C/" + rv_group], RID_A], axis=1
                )
                demand_sample_ext["D/" + rv_group] = pd.concat(
                    [demand_sample["D/" + rv_group], RID_B], axis=1
                )
            else:
                demand_sample_ext["C/" + rv_group] = pd.concat(
                    [demand_sample["C/" + rv_group], RID_B], axis=1
                )
                demand_sample_ext["D/" + rv_group] = pd.concat(
                    [demand_sample["D/" + rv_group], RID_A], axis=1
                )

        # add spectral acceleration
        spectrum = pd.read_csv(
            f"results/site_hazard/UHS_{hz}.csv", index_col=0, header=0
        )
        ifun = interp1d(spectrum.index.to_numpy(), spectrum.to_numpy().reshape(-1))
        base_period = float(read_study_param(f"data/{archetype}/period"))
        sa_t = float(ifun(base_period))
        for dmse in demand_sample_ext.values():
            dmse[("SA", "0", "1")] = sa_t
            # add units to the data
            dmse.T.insert(0, "Units", "")
            dmse.loc["Units", ["PFA", "SA"]] = "g"
            dmse.loc["Units", ["PID", "RID"]] = "rad"
            dmse.loc["Units", ["PFV"]] = "inps2"
            dmse.loc["Units", ["SA"]] = "g"
        # load back the demand sample
        for key, asmt in asmts.items():
            asmt.demand.load_sample(demand_sample_ext[key])

    demand_add_rid_and_sa()

    # ---------------------------------- #
    # Damage Estimation                  #
    # ---------------------------------- #

    def damage_load_component_configuration():
        """
        load the component configuration
        """
        cmp_marginals_structural = pd.read_csv(
            f"data/{archetype}/performance/input_cmp_quant.csv", index_col=0
        )
        cmp_marginals_nonstructural = pd.read_csv(
            f"data/performance/input_cmp_quant_{occupancy}_{num_stories}.csv",
            index_col=0,
        )
        cmp_marginals_building = pd.DataFrame(
            {
                "Units": ["ea", "ea", "ea"],
                "Location": ["all", "0", "0"],
                "Direction": ["1,2", "1", "1"],
                "Theta_0": ["1", "1", "1"],
            },
            index=["excessiveRID", "collapse", "irreparable"],
        )
        cmp_marginals = pd.concat(
            (
                cmp_marginals_structural,
                cmp_marginals_nonstructural,
                cmp_marginals_building,
            )
        )

        for asmt in asmts.values():
            asmt.asset.load_cmp_model({"marginals": cmp_marginals})

        return cmp_marginals

    cmp_marginals = damage_load_component_configuration()

    def damage_generate_sample():
        """
        Generate the component quantity sample
        """
        for key in ("A", "B"):
            asmt = asmts[key]
            asmt.asset.generate_cmp_sample()

    damage_generate_sample()

    def damage_transfer_rvs():
        """
        transfer RVs to assessments C and D
        """
        for rv_group in rv_groups:
            asmts["C/" + rv_group].asset._cmp_RVs = uq.RandomVariableRegistry(
                asmts["C/" + rv_group].options.rng
            )
            asmts["D/" + rv_group].asset._cmp_RVs = uq.RandomVariableRegistry(
                asmts["D/" + rv_group].options.rng
            )

        rv_names = asmts["A"].asset._cmp_RVs.RV.keys()
        assert rv_names == asmts["B"].asset._cmp_RVs.RV.keys()

        for rv_group in rv_groups:
            rv_name_filter_tags = rv_filtering[rv_group]["include"]
            rv_name_filter_tag_exclude = rv_filtering[rv_group]["exclude"]
            assert isinstance(rv_name_filter_tags, tuple)
            assert isinstance(rv_name_filter_tag_exclude, tuple)
            for rv_name in rv_names:
                # if it matches from A, it goes to C,
                # otherwise it goes to D.
                if matches(rv_name, rv_name_filter_tags, rv_name_filter_tag_exclude):
                    asmts["C/" + rv_group].asset._cmp_RVs.add_RV(
                        asmts["A"].asset._cmp_RVs.RV[rv_name]
                    )
                    asmts["D/" + rv_group].asset._cmp_RVs.add_RV(
                        asmts["B"].asset._cmp_RVs.RV[rv_name]
                    )
                else:
                    asmts["C/" + rv_group].asset._cmp_RVs.add_RV(
                        asmts["B"].asset._cmp_RVs.RV[rv_name]
                    )
                    asmts["D/" + rv_group].asset._cmp_RVs.add_RV(
                        asmts["A"].asset._cmp_RVs.RV[rv_name]
                    )

    damage_transfer_rvs()

    def damage_define_damage_model():
        """
        define component fragilities
        """

        damage_db = pd.read_csv(
            "data/performance/input_damage.csv", header=[0, 1], index_col=0
        )

        damage_db.rename(columns=rm_unnamed, level=1, inplace=True)

        # update collapse fragility
        bldg_fragilities = pd.read_csv(
            "results/response/collapse_fragilities.csv", header=0, index_col=[0]
        )
        vals = bldg_fragilities.loc[
            " ".join([x.upper() for x in archetype.split("_")])
        ]
        damage_db.loc[
            "collapse", [("LS1", "Family"), ("LS1", "Theta_0"), ("LS1", "Theta_1")]
        ] = ["lognormal", vals.Median, vals.Beta]

        for asmt in asmts.values():
            asmt.damage.load_damage_model([damage_db])

    damage_define_damage_model()

    def damage_define_component_capacity():
        """
        define the component capacities
        """

        # PAL.damage.calculate(dmg_process=dmg_process)#, block_batch_size=100)
        # damage_sample = PAL.damage.save_sample()

        pg_batch = asmts["A"].damage._get_pg_batches(10000000)
        pd.testing.assert_frame_equal(
            pg_batch, asmts["B"].damage._get_pg_batches(10000000)
        )
        for rv_group in rv_groups:
            pd.testing.assert_frame_equal(
                pg_batch, asmts["C/" + rv_group].damage._get_pg_batches(10000000)
            )
            pd.testing.assert_frame_equal(
                pg_batch, asmts["D/" + rv_group].damage._get_pg_batches(10000000)
            )
            PGB = pg_batch.xs(1, level=0, axis=0)

        # capacity_sample, lsds_sample = asmts['A'].damage._generate_dmg_sample(
        #     num_realizations, PGB)
        # breaking down...

        # Create capacity and LSD RVs for each performance group
        capacity_RVs = {}
        lsds_RVs = {}
        capacity_RVs["A"], lsds_RVs["A"] = asmts["A"].damage._create_dmg_RVs(PGB)
        capacity_RVs["B"], lsds_RVs["B"] = asmts["B"].damage._create_dmg_RVs(PGB)

        # Generate samples for capacity and LSDS RVs
        for key in ("A", "B"):
            capacity_RVs[key].generate_sample(
                sample_size=num_realizations,
                method=asmts["A"].damage._asmnt.options.sampling_method,
            )
            lsds_RVs[key].generate_sample(
                sample_size=num_realizations,
                method=asmts["A"].damage._asmnt.options.sampling_method,
            )

        return PGB, capacity_RVs, lsds_RVs

    PGB, capacity_RVs, lsds_RVs = damage_define_component_capacity()

    def damage_initialize_capacity_lsds():
        """
        initialize capacity lsds
        """
        for rv_group in rv_groups:
            capacity_RVs["C/" + rv_group] = uq.RandomVariableRegistry(
                asmts["C/" + rv_group].options.rng
            )
            capacity_RVs["D/" + rv_group] = uq.RandomVariableRegistry(
                asmts["D/" + rv_group].options.rng
            )
            lsds_RVs["C/" + rv_group] = uq.RandomVariableRegistry(
                asmts["C/" + rv_group].options.rng
            )
            lsds_RVs["D/" + rv_group] = uq.RandomVariableRegistry(
                asmts["D/" + rv_group].options.rng
            )

    damage_initialize_capacity_lsds()

    def damage_transfer_component_capacity():
        # transfer RVs to assessments C and D
        rv_names = capacity_RVs["A"].RV.keys()
        assert rv_names == capacity_RVs["B"].RV.keys()

        for rv_group in rv_groups:
            rv_name_filter_tags = rv_filtering[rv_group]["include"]
            rv_name_filter_tag_exclude = rv_filtering[rv_group]["exclude"]
            assert isinstance(rv_name_filter_tags, tuple)
            assert isinstance(rv_name_filter_tag_exclude, tuple)

            for rv_name in rv_names:
                # if it matches from A, it goes to C,
                # otherwise it goes to D.
                if matches(rv_name, rv_name_filter_tags, rv_name_filter_tag_exclude):
                    capacity_RVs["C/" + rv_group].add_RV(
                        capacity_RVs["A"].RV[rv_name]
                    )
                    capacity_RVs["D/" + rv_group].add_RV(
                        capacity_RVs["B"].RV[rv_name]
                    )
                else:
                    capacity_RVs["C/" + rv_group].add_RV(
                        capacity_RVs["B"].RV[rv_name]
                    )
                    capacity_RVs["D/" + rv_group].add_RV(
                        capacity_RVs["A"].RV[rv_name]
                    )

    damage_transfer_component_capacity()

    def damage_transfer_component_lsds():
        """
        transfer component lsds
        """
        rv_names = lsds_RVs["A"].RV.keys()
        assert rv_names == lsds_RVs["B"].RV.keys()

        for rv_group in rv_groups:
            rv_name_filter_tags = rv_filtering[rv_group]["include"]
            rv_name_filter_tag_exclude = rv_filtering[rv_group]["exclude"]
            assert isinstance(rv_name_filter_tags, tuple)
            assert isinstance(rv_name_filter_tag_exclude, tuple)

            for rv_name in rv_names:
                # if it matches from A, it goes to C,
                # otherwise it goes to D.
                if matches(rv_name, rv_name_filter_tags, rv_name_filter_tag_exclude):
                    lsds_RVs["C/" + rv_group].add_RV(lsds_RVs["A"].RV[rv_name])
                    lsds_RVs["D/" + rv_group].add_RV(lsds_RVs["B"].RV[rv_name])
                else:
                    lsds_RVs["C/" + rv_group].add_RV(lsds_RVs["B"].RV[rv_name])
                    lsds_RVs["D/" + rv_group].add_RV(lsds_RVs["A"].RV[rv_name])

    damage_transfer_component_lsds()

    def damage_get_capacity_lsds_samples():
        """
        get the capacity and lsds samples
        """

        capacity_sample = {}
        lsds_sample = {}
        for key in asmts:
            capacity_sample[key] = (
                pd.DataFrame(capacity_RVs[key].RV_sample)
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
            capacity_sample[key] = base.convert_to_MultiIndex(
                capacity_sample[key], axis=1
            )["FRG"]
            capacity_sample[key].columns.names = [
                "cmp",
                "loc",
                "dir",
                "uid",
                "block",
                "ls",
            ]

            lsds_sample[key] = (
                pd.DataFrame(lsds_RVs[key].RV_sample)
                .sort_index(axis=0)
                .sort_index(axis=1)
                .astype(int)
            )
            lsds_sample[key] = base.convert_to_MultiIndex(lsds_sample[key], axis=1)[
                "LSDS"
            ]
            lsds_sample[key].columns.names = [
                "cmp",
                "loc",
                "dir",
                "uid",
                "block",
                "ls",
            ]

        return capacity_sample, lsds_sample

    capacity_sample, lsds_sample = damage_get_capacity_lsds_samples()

    def damage_calculate():
        """
        calculate damage
        """
        # Now perform the rest of the deterministic operations that result in
        # the assignment of .damage._sample

        for key, asmt in asmts.items():
            EDP_req = asmt.damage._get_required_demand_type(PGB)
            demand_dict = asmt.damage._assemble_required_demand_data(EDP_req)
            ds_sample = (
                asmt.damage._evaluate_damage_state(  # deterministic operation
                    demand_dict, EDP_req, capacity_sample[key], lsds_sample[key]
                )
            )

            # Store the damage state sample as a local variable
            dmg_ds = ds_sample

            # apply flooding damage process
            dmg_ds.loc[:, "C.30.21.001k"].to_numpy()[
                np.where(dmg_ds.loc[:, "D.20.21.013a"] == 2.0)
            ] = 1.0
            dmg_ds.loc[:, "C.30.21.001k"].to_numpy()[
                np.where(dmg_ds.loc[:, "D.20.31.013b"] == 1.0)
            ] = 1.0
            dmg_ds.loc[:, "C.30.21.001k"].to_numpy()[
                np.where(dmg_ds.loc[:, "D.40.11.024a"] == 2.0)
            ] = 1.0

            # Retrieve the component quantity information from the asset
            # model
            cmp_qnt = asmt.damage._asmnt.asset.cmp_sample  # .values
            # Retrieve the component marginal parameters from the asset
            # model
            cmp_params = asmt.damage._asmnt.asset.cmp_marginal_params

            # Combine the component quantity information for the columns
            # in the damage state sample
            dmg_qnt = pd.concat(
                [cmp_qnt[PG[:4]] for PG in dmg_ds.columns],
                axis=1,
                keys=dmg_ds.columns,
            )

            # Initialize a list to store the block weights
            block_weights = []

            # For each component in the list of PG blocks
            for PG in PGB.index:
                # Set the number of blocks to 1, unless specified
                # otherwise in the component marginal parameters
                blocks = 1
                if cmp_params is not None:
                    if "Blocks" in cmp_params.columns:
                        blocks = cmp_params.loc[PG, "Blocks"]

                # If the number of blocks is specified, calculate the
                # weights as the reciprocal of the number of blocks
                if np.atleast_1d(blocks).shape[0] == 1:
                    blocks_array = np.full(int(blocks), 1.0 / blocks)

                # Otherwise, assume that the list contains the weights
                block_weights += blocks_array.tolist()

            # Broadcast the block weights to match the shape of the damage
            # quantity DataFrame
            block_weights = np.broadcast_to(
                block_weights, (dmg_qnt.shape[0], len(block_weights))
            )

            # Multiply the damage quantities by the block weights
            dmg_qnt *= block_weights

            # # Get the unique damage states from the damage state sample
            # # Note that these might be fewer than all possible Damage
            # # States
            # ds_list = np.unique(dmg_ds.values)
            # # Filter out any NaN values from the list of damage states
            # ds_list = ds_list[pd.notna(ds_list)].astype(int)

            # ^^^ we don't want this for VBSA. We need to include all possible
            # damage states.

            # but instead of using the same list for all components, it would be
            # better to consider the number of valid damage states of each.

            # ds_list = np.arange(16, dtype=float)

            dmg_params = asmt.damage.damage_params.T.to_dict(orient="series")
            num_ds = {}
            num_ls = {}
            for comp in dmg_params:
                th0s = dmg_params[comp].xs("Theta_0", level=1).dropna()
                n_ds = len(th0s)
                num_ds[comp] = n_ds
                dsws = (
                    dmg_params[comp]
                    .loc[th0s.index]
                    .xs("DamageStateWeights", level=1)
                    .fillna("1.00")
                    .to_list()
                )
                n_ls = 0
                for x in dsws:
                    n_ls += len(x.split("|"))
                    num_ls[comp] = n_ls

            # # If the dropzero option is True, remove the zero damage state
            # # from the list of damage states
            # if dropzero:

            #     ds_list = ds_list[ds_list != 0]

            res_dfs = []
            dmg_qnt_dct = dmg_qnt.to_dict(orient="series")
            dmg_ds_dct = dmg_ds.to_dict(orient="series")
            for entry in dmg_qnt_dct:
                n_ds = num_ls[entry[0]]
                for ds in range(n_ds):
                    res_dfs.append(
                        pd.Series(
                            np.where(
                                dmg_ds_dct[entry] == ds + 1, dmg_qnt_dct[entry], 0
                            ),
                            name=(*entry, f"{ds+1}"),
                        )
                    )
            res_df = pd.concat(res_dfs, axis=1)
            res_df.columns.names = ("cmp", "loc", "dir", "uid", "block", "ds")

            # # Only proceed with the calculation if there is at least one
            # # damage state in the list
            # if len(ds_list) > 0:

            # # Create a list of DataFrames, where each DataFrame stores
            # # the damage quantities for a specific damage state

            # res_list = [pd.DataFrame(
            #     np.where(dmg_ds == ds_i, dmg_qnt, 0),
            #     columns=dmg_ds.columns,
            #     index=dmg_ds.index
            # ) for ds_i in ds_list]

            # # Combine the damage quantity DataFrames into a single
            # # DataFrame
            # res_df = pd.concat(
            #     res_list, axis=1,
            #     keys=[f'{ds_i:g}' for ds_i in ds_list])
            # res_df.columns.names = ['ds', *res_df.columns.names[1::]]
            # # remove the block level from the columns
            # res_df.columns = res_df.columns.reorder_levels([1, 2, 3, 4, 0, 5])

            res_df = res_df.groupby(level=[0, 1, 2, 3, 5], axis=1).sum()

            # # The damage states with no damaged quantities are dropped
            # # Note that some of these are not even valid DSs at the given PG
            # res_df = res_df.iloc[:, np.where(res_df.sum(axis=0) != 0)[0]]

            qnt_sample = res_df
            # end breaking down: _prepare_dmg_quantities

            # qnt_samples.append(qnt_sample)
            # qnt_sample = pd.concat(qnt_samples, axis=1)
            qnt_sample.sort_index(axis=1, inplace=True)

            # deterministic operations..
            dmg_process = {
                "1_collapse": {"DS1": "ALL_NA"},
                "2_excessiveRID": {"DS1": "irreparable_DS1"},
            }

            if dmg_process is not None:
                dmg_process = {key: dmg_process[key] for key in sorted(dmg_process)}
                for task in dmg_process.items():
                    qnt_sample = asmt.damage._perform_dmg_task(task, qnt_sample)
                    # if asmt.damage._dmg_function_scale_factors is not None:
                    #     qnt_sample = asmt.damage._apply_damage_functions(
                    #         EDP_req, demand, qnt_sample
                    #     )

            asmt.damage._sample = qnt_sample

    damage_calculate()

    # ---------------------------------- #
    # Loss Estimation                    #
    # ---------------------------------- #

    def loss_construct_loss_map():
        drivers = [f"DMG-{cmp}" for cmp in cmp_marginals.index.unique()]
        drivers = drivers[:-3] + drivers[-2:]
        loss_models = cmp_marginals.index.unique().tolist()[:-3]
        loss_models += [
            "replacement",
        ] * 2
        loss_map = pd.DataFrame(loss_models, columns=["BldgRepair"], index=drivers)
        return loss_map

    loss_map = loss_construct_loss_map()

    def loss_obtain_loss_db():
        loss_db = pd.read_csv(
            "data/performance/input_loss.csv", header=[0, 1], index_col=[0, 1]
        )
        loss_db.rename(columns=rm_unnamed, level=1, inplace=True)
        loss_db_additional = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                (
                    ("replacement", "Carbon"),
                    ("replacement", "Cost"),
                    ("replacement", "Energy"),
                    ("replacement", "Time"),
                )
            ),
            columns=loss_db.columns,
        )
        loss_db_additional.loc["replacement", ("DV", "Unit")] = [
            "kg",
            "USD_2011",
            "MJ",
            "worker_day",
        ]
        loss_db_additional.loc["replacement", ("Quantity", "Unit")] = ["1 EA"] * 4
        bldg_repl_df = pd.read_csv(
            "data/performance/bldg_replacement.csv", header=0, index_col=[0, 1]
        )
        loss_db_additional.loc[("replacement", "Cost"), "DS1"] = [
            "lognormal",
            np.nan,
            bldg_repl_df.at[(num_stories, occupancy), "replacement_cost"]
            * b_repl_factor,
            0.35,
        ]
        loss_db_additional.loc[("replacement", "Time"), "DS1"] = [
            "lognormal",
            np.nan,
            bldg_repl_df.at[(num_stories, occupancy), "replacement_time"]
            * b_repl_factor,
            0.35,
        ]
        loss_db = loss_db._append(loss_db_additional)
        replacement_cost = loss_db_additional.loc[
            ("replacement", "Cost"), ("DS1", "Theta_0")
        ]
        return loss_db, replacement_cost

    loss_db, replacement_cost = loss_obtain_loss_db()

    def loss_load_model():
        """
        load loss model
        """
        asmts["A"].bldg_repair.load_model([loss_db], loss_map)
        for key, asmt in asmts.items():
            if key == "A":
                continue
            asmt.bldg_repair.loss_map = asmts["A"].bldg_repair.loss_map
            asmt.bldg_repair.loss_params = asmts["A"].bldg_repair.loss_params

    loss_load_model()

    # asmts['A'].bldg_repair.calculate()
    # breaking down...

    def loss_generate_sample():
        # First, get the damaged quantities in each damage state for
        # each component of interest.
        dmg_quantities = {}
        for key, asmt in asmts.items():
            dmg_quantities[key] = asmt.damage.sample

        # asmts['A'].bldg_repair._generate_DV_sample(dmg_q, sample_size)
        # breaking down ...

        # calculate the quantities for economies of scale

        if asmts["A"].options.eco_scale["AcrossFloors"] == True:
            if asmts["A"].options.eco_scale["AcrossDamageStates"] == True:
                eco_levels = [
                    0,
                ]
                eco_columns = [
                    "cmp",
                ]
            else:
                eco_levels = [0, 4]
                eco_columns = ["cmp", "ds"]
        elif asmts["A"].options.eco_scale["AcrossDamageStates"] == True:
            eco_levels = [0, 1]
            eco_columns = ["cmp", "loc"]
        else:
            eco_levels = [0, 1, 4]
            eco_columns = ["cmp", "loc", "ds"]

        eco_qnt = {}
        for key, asmt in asmts.items():
            eco_group = dmg_quantities[key].groupby(level=eco_levels, axis=1)
            eco_qnt[key] = eco_group.sum().mask(eco_group.count() == 0, np.nan)
            assert eco_qnt[key].columns.names == eco_columns

        # eco_qnt["A"]
        # eco_qnt["B"]
        # eco_qnt["C"]
        # eco_qnt["D"]

        # apply the median functions, if needed, to get median consequences for
        # each realization
        medians = {}
        for key, asmt in asmts.items():
            medians[key] = asmts[key].bldg_repair._calc_median_consequence(
                eco_qnt[key]
            )
            # note: the above considers the different consequences for components
            # that have many.
            # medians["A"]["Cost"]
            # medians["B"]["Cost"]
            # medians["C"]["Cost"]
            # medians["D"]["Cost"]

        # combine the median consequences with the samples of deviation from the
        # median to get the consequence realizations.
        loss_RV_reg = {}
        loss_RV_reg["A"] = asmts["A"].bldg_repair._create_DV_RVs(
            dmg_quantities["A"].columns
        )
        loss_RV_reg["A"].generate_sample(
            sample_size=num_realizations,
            method=asmts["A"].bldg_repair._asmnt.options.sampling_method,
        )
        loss_RV_reg["B"] = asmts["B"].bldg_repair._create_DV_RVs(
            dmg_quantities["B"].columns
        )
        loss_RV_reg["B"].generate_sample(
            sample_size=num_realizations,
            method=asmts["B"].bldg_repair._asmnt.options.sampling_method,
        )

        return loss_RV_reg, dmg_quantities, medians

    loss_RV_reg, dmg_quantities, medians = loss_generate_sample()

    def loss_transfer_rvs():
        """
        transfer RVs to assessments C and D
        """

        def loss_RV_id_to_cmp_name(original_rv_name):
            """
            Converts the original_rv_name to a new name by replacing the
            second part (separated by "-") with the corresponding cmap_name
            from the loss_map DataFrame using the cmp_id. Returns the modified
            name.

            Parameters:
            original_rv_name (str): The original RV name to be modified.

            Returns:
            str: The modified RV name.
            """
            parts = original_rv_name.split("-")
            cmp_id = parts[1]
            cmp_name = loss_map.iloc[int(cmp_id)].values[0]
            parts[1] = cmp_name
            return "-".join(parts)

        rv_names = loss_RV_reg["A"].RV.keys()
        assert rv_names == loss_RV_reg["B"].RV.keys()

        for rv_group in rv_groups:
            rv_name_filter_tags = rv_filtering[rv_group]["include"]
            rv_name_filter_tag_exclude = rv_filtering[rv_group]["exclude"]
            assert isinstance(rv_name_filter_tags, tuple)
            assert isinstance(rv_name_filter_tag_exclude, tuple)

            loss_RV_reg["C/" + rv_group] = uq.RandomVariableRegistry(
                asmts["C/" + rv_group].options.rng
            )
            loss_RV_reg["D/" + rv_group] = uq.RandomVariableRegistry(
                asmts["D/" + rv_group].options.rng
            )

            for rv_name in rv_names:
                # if it matches from A, it goes to C,
                # otherwise it goes to D.
                if matches(
                    loss_RV_id_to_cmp_name(rv_name),
                    rv_name_filter_tags,
                    rv_name_filter_tag_exclude,
                ):
                    loss_RV_reg["C/" + rv_group].add_RV(loss_RV_reg["A"].RV[rv_name])
                    loss_RV_reg["D/" + rv_group].add_RV(loss_RV_reg["B"].RV[rv_name])
                else:
                    loss_RV_reg["C/" + rv_group].add_RV(loss_RV_reg["B"].RV[rv_name])
                    loss_RV_reg["D/" + rv_group].add_RV(loss_RV_reg["A"].RV[rv_name])

        loss_std_sample = {}
        for key in asmts:
            loss_std_sample[key] = base.convert_to_MultiIndex(
                pd.DataFrame(loss_RV_reg[key].RV_sample), axis=1
            ).sort_index(axis=1)
            loss_std_sample[key].columns.names = [
                "dv",
                "cmp",
                "ds",
                "loc",
                "dir",
                "uid",
            ]
            std_idx = loss_std_sample[key].columns.levels
            loss_std_sample[key].columns = loss_std_sample[key].columns.set_levels(
                [
                    std_idx[0],
                    std_idx[1].astype(int),
                    std_idx[2],
                    std_idx[3],
                    std_idx[4],
                    std_idx[5],
                ]
            )
            loss_std_sample[key].sort_index(axis=1, inplace=True)

        return loss_std_sample

    loss_std_sample = loss_transfer_rvs()

    def loss_calculate():
        """
        calculate losses
        """
        for key, asmt in asmts.items():
            res_list = []
            key_list = []

            dmg_quantities[key].columns = dmg_quantities[key].columns.reorder_levels(
                [0, 4, 1, 2, 3]
            )
            dmg_quantities[key].sort_index(axis=1, inplace=True)

            # DV_types = asmt.bldg_repair.loss_params.index.unique(level=1)
            DV_types = ["Cost", "Time"]

            # std_DV_types = std_sample[key].columns.unique(level=0)

            for DV_type in DV_types:
                prob_cmp_list = loss_std_sample[key][DV_type].columns.unique(level=0)

                cmp_list = []

                assert DV_type in medians[key]

                for cmp_i in medians[key][DV_type].columns.unique(level=0):
                    # check if there is damage in the component
                    driver_type, dmg_cmp_i = asmt.bldg_repair.loss_map.loc[
                        cmp_i, "Driver"
                    ]
                    loss_cmp_i = asmt.bldg_repair.loss_map.loc[cmp_i, "Consequence"]

                    if driver_type != "DMG":
                        raise ValueError(
                            f"Loss Driver type not " f"recognized: {driver_type}"
                        )

                    assert dmg_cmp_i in dmg_quantities[key].columns.unique(level=0)

                    ds_list = []

                    for ds in (
                        medians[key][DV_type].loc[:, cmp_i].columns.unique(level=0)
                    ):
                        loc_list = []

                        for loc_id, loc in enumerate(
                            dmg_quantities[key]
                            .loc[:, (dmg_cmp_i, ds)]
                            .columns.unique(level=0)
                        ):
                            if (asmt.options.eco_scale["AcrossFloors"] is True) and (
                                loc_id > 0
                            ):
                                break

                            if asmt.options.eco_scale["AcrossFloors"] is True:
                                median_i = medians[key][DV_type].loc[:, (cmp_i, ds)]
                                dmg_i = dmg_quantities[key].loc[:, (dmg_cmp_i, ds)]

                                if cmp_i in prob_cmp_list:
                                    std_i = loss_std_sample[key].loc[
                                        :, (DV_type, cmp_i, ds)
                                    ]
                                else:
                                    std_i = None

                            else:
                                median_i = medians[key][DV_type].loc[
                                    :, (cmp_i, ds, loc)
                                ]
                                dmg_i = dmg_quantities[key].loc[
                                    :, (dmg_cmp_i, ds, loc)
                                ]

                                if cmp_i in prob_cmp_list:
                                    std_i = loss_std_sample[key].loc[
                                        :, (DV_type, cmp_i, ds, loc)
                                    ]
                                else:
                                    std_i = None

                            if std_i is not None:
                                res_list.append(dmg_i.mul(median_i, axis=0) * std_i)
                            else:
                                res_list.append(dmg_i.mul(median_i, axis=0))

                            loc_list.append(loc)

                        if asmt.options.eco_scale["AcrossFloors"] is True:
                            ds_list += [
                                ds,
                            ]
                        else:
                            ds_list += [(ds, loc) for loc in loc_list]

                    if asmt.options.eco_scale["AcrossFloors"] is True:
                        cmp_list += [(loss_cmp_i, dmg_cmp_i, ds) for ds in ds_list]
                    else:
                        cmp_list += [
                            (loss_cmp_i, dmg_cmp_i, ds, loc) for ds, loc in ds_list
                        ]

                if asmt.options.eco_scale["AcrossFloors"] is True:
                    key_list += [
                        (DV_type, loss_cmp_i, dmg_cmp_i, ds)
                        for loss_cmp_i, dmg_cmp_i, ds in cmp_list
                    ]
                else:
                    key_list += [
                        (DV_type, loss_cmp_i, dmg_cmp_i, ds, loc)
                        for loss_cmp_i, dmg_cmp_i, ds, loc in cmp_list
                    ]

            lvl_names = ["dv", "loss", "dmg", "ds", "loc", "dir", "uid"]
            DV_sample = pd.concat(res_list, axis=1, keys=key_list, names=lvl_names)

            DV_sample = DV_sample.fillna(0).convert_dtypes()
            DV_sample.columns.names = lvl_names

            std_idx = DV_sample.columns.levels
            DV_sample.columns = DV_sample.columns.set_levels(
                [
                    std_idx[0],
                    std_idx[1],
                    std_idx[2],
                    std_idx[3],
                    std_idx[4].astype(int),
                    std_idx[5],
                    std_idx[6],
                ]
            )

            # Get the flags for replacement consequence trigger
            DV_sum = DV_sample.groupby(
                level=[
                    1,
                ],
                axis=1,
            ).sum()
            if "replacement" in DV_sum.columns:
                # When the 'replacement' consequence is triggered, all
                # local repair consequences are discarded. Note that
                # global consequences are assigned to location '0'.

                id_replacement = DV_sum["replacement"] > 0

                # get the list of non-zero locations
                locs = DV_sample.columns.get_level_values(4).unique().values

                locs = locs[locs != 0]

                idx = pd.IndexSlice
                DV_sample.loc[id_replacement, idx[:, :, :, :, locs]] = 0.0

            asmt.bldg_repair._sample = DV_sample

    loss_calculate()

    def loss_aggregate():
        """
        aggregate losses
        """

        for total_loss_threshold in total_loss_thresholds:
            agg_df = {}

            for key, asmt in asmts.items():
                DV = asmt.bldg_repair.sample

                # determine non-replacement realizations
                cases = DV["Cost"].loc[:, "replacement"].sum(axis=1).astype(int)
                case_idxs = cases[cases == 0].index

                # determine replacement threshold
                repl_threshold = replacement_cost * total_loss_threshold
                # determine realizations exceeding the threshold
                sums = DV["Cost"].loc[case_idxs, :].sum(axis=1)
                exc_idxs = sums[sums > repl_threshold].index

                pr_repl = len(set(case_idxs).union(set(exc_idxs))) / num_realizations
                if key == "A":
                    with open(
                        store_info(
                            f"results/risk/{output_dir_name}/"
                            f"replacementProb_{out_prefix}_"
                            f"{archetype}_"
                            f"{occupancy}_{modeling_uncertainty}_"
                            f"{total_loss_threshold}_"
                            f"{b_repl_factor}_"
                            f"{hz}_{repetition_counter+1}.txt"
                        ),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(str(pr_repl))

                # group results by DV type and location
                DVG = DV.groupby(level=[0, 4], axis=1).sum()

                # create the summary DF
                df_agg = pd.DataFrame(
                    index=DV.index,
                    columns=[
                        "repair_cost",
                        "repair_time-parallel",
                        "repair_time-sequential",
                        "repair_carbon",
                        "repair_energy",
                    ],
                )

                if "Cost" in DVG.columns:
                    df_agg["repair_cost"] = DVG["Cost"].sum(axis=1)
                    # consider total loss threshold
                    if len(df_agg["repair_cost"].loc[exc_idxs]) > 0:
                        df_agg.loc[exc_idxs, "repair_cost"] = loss_std_sample[key][
                            "Cost"
                        ].iloc[exc_idxs, -1]
                else:
                    df_agg = df_agg.drop("repair_cost", axis=1)

                if "Time" in DVG.columns:
                    df_agg["repair_time-sequential"] = DVG["Time"].sum(axis=1)
                    df_agg["repair_time-parallel"] = DVG["Time"].max(axis=1)
                    # consider total loss threshold
                    if len(df_agg["repair_cost"].loc[exc_idxs]) > 0:
                        for thing in (
                            "repair_time-sequential",
                            "repair_time-parallel",
                        ):
                            df_agg.loc[exc_idxs, thing] = loss_std_sample[key][
                                "Time"
                            ].iloc[exc_idxs, -1]
                else:
                    df_agg = df_agg.drop(
                        ["repair_time-parallel", "repair_time-sequential"], axis=1
                    )

                if "Carbon" in DVG.columns:
                    df_agg["repair_carbon"] = DVG["Carbon"].sum(axis=1)
                else:
                    df_agg = df_agg.drop("repair_carbon", axis=1)

                if "Energy" in DVG.columns:
                    df_agg["repair_energy"] = DVG["Energy"].sum(axis=1)
                else:
                    df_agg = df_agg.drop("repair_energy", axis=1)

                # # convert units
                # cmp_units = (
                #     asmt.bldg_repair.loss_params[("DV", "Unit")]
                #     .groupby(
                #         level=[
                #             1,
                #         ]
                #     )
                #     .agg(lambda x: x.value_counts().index[0])
                # )
                # dv_units = pd.Series(index=df_agg.columns, name="Units", dtype="object")
                # if "Cost" in DVG.columns:
                #     dv_units["repair_cost"] = cmp_units["Cost"]
                # if "Time" in DVG.columns:
                #     dv_units["repair_time-parallel"] = cmp_units["Time"]
                #     dv_units["repair_time-sequential"] = cmp_units["Time"]
                # if "Carbon" in DVG.columns:
                #     dv_units["repair_carbon"] = cmp_units["Carbon"]
                # if "Energy" in DVG.columns:
                #     dv_units["repair_energy"] = cmp_units["Energy"]
                # df_agg = file_io.save_to_csv(
                #     df_agg,
                #     None,
                #     units=dv_units,
                #     unit_conversion_factors=(
                #         asmt.bldg_repair._asmnt.unit_conversion_factors,
                #     ),
                #     use_simpleindex=False,
                #     log=asmt.bldg_repair._asmnt.log,
                # )
                # df_agg.drop("Units", inplace=True)

                # convert header
                df_agg = base.convert_to_MultiIndex(df_agg, axis=1).astype(float)

                agg_df[key] = df_agg

            for loss_type in loss_types:
                ys = {}
                if loss_type == "Cost":
                    for key in asmts:
                        ys[key] = agg_df[key]["repair_cost"]

                elif loss_type == "Time":
                    for key in asmts:
                        ys[key] = agg_df[key]["repair_time"]["sequential"]

                else:
                    raise ValueError(f"Unsupported loss type: {loss_type}")

                y_mat = pd.concat(ys.values(), axis=1, keys=ys.keys())

                y_mat.to_parquet(
                    store_info(
                        f"results/risk/{output_dir_name}/{out_prefix}_{archetype}_"
                        f"{occupancy}_{modeling_uncertainty}_"
                        f"{b_repl_factor}_"
                        f"{total_loss_threshold}_{loss_type}_"
                        f"{hz}_{repetition_counter+1}.parquet"
                    ),
                    index=None,
                )

    loss_aggregate()


if __name__ == "__main__":

    for repetition_counter in range(num_repetitions[0], num_repetitions[1]):
        main(repetition_counter)
