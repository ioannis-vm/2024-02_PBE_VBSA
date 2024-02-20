"""
Run nonlinear time-history analysis to get the building's response
"""

import os
import importlib
import argparse
import numpy as np
import pandas as pd
from osmg import solver
from osmg.ground_motion_utils import import_PEER
from src.util import read_study_param
from src.util import retrieve_peer_gm_data
from src.util import store_info

# ~~~~~~~~~~~~~~~~~~~~~~ #
# set up argument parser #
# ~~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument("--archetype")
parser.add_argument("--hazard_level")
parser.add_argument("--gm_number")
parser.add_argument("--analysis_dt")
parser.add_argument("--output_dir_name", default="response_modal")
parser.add_argument("--progress_bar", default=False)
parser.add_argument("--custom_path", default=None)
parser.add_argument("--damping", default="modal")

args = parser.parse_args()
archetype = args.archetype
hazard_level = args.hazard_level
gm_number = int(args.gm_number)
analysis_dt = float(args.analysis_dt)
output_dir_name = args.output_dir_name
progress_bar = bool(args.progress_bar)
custom_path = args.custom_path
damping = args.damping

# archetype = 'smrf_3_ii'
# hazard_level = '8'
# gm_number = 3
# analysis_dt = 0.01
# output_dir_name = 'response_modal'
# progress_bar = True
# custom_path = 'tmp/test/all_fixed'

# load archetype building
archetypes_module = importlib.import_module("src.structural_analysis.archetypes")
try:
    archetype_builder = getattr(archetypes_module, archetype)
except AttributeError as exc:
    raise ValueError(f"Invalid archetype code: {archetype}") from exc

mdl, loadcase = archetype_builder()

# from osmg.graphics.preprocessing_3d import show
# show(mdl, loadcase, extrude=True)

num_levels = len(mdl.levels) - 1
level_heights = []
for level in mdl.levels.values():
    level_heights.append(level.elevation)
level_heights = np.diff(level_heights)

lvl_nodes = []
base_node = list(mdl.levels[0].nodes.values())[0].uid
lvl_nodes.append(base_node)
for i in range(num_levels):
    lvl_nodes.append(loadcase.parent_nodes[i + 1].uid)

specific_nodes = lvl_nodes + [n.uid for n in mdl.levels[0].nodes.values()]

df_records = pd.read_csv(
    "results/site_hazard/required_records_and_scaling_factors_adjusted_to_cms.csv",
    index_col=[0, 1, 2],
)

record_trim = pd.read_csv("results/site_hazard/record_trim.csv", index_col=0)

rsn = int(df_records.at[(archetype, f"hz_{hazard_level}", "RSN"), str(gm_number)])

finish_time = record_trim.at[rsn, "0"]
if pd.isna(finish_time):
    finish_time = 0.00  # 0.00 means run the whole record

scaling = df_records.at[(archetype, f"hz_{hazard_level}", "SF"), str(gm_number)]

gm_filenames = retrieve_peer_gm_data(rsn)

gm_data_x = import_PEER(gm_filenames[0])
gm_data_y = import_PEER(gm_filenames[1])
if gm_filenames[2]:
    gm_data_z = import_PEER(gm_filenames[2])
else:
    gm_data_z = None

# ensure that the time-histories have the same ground motion dt
gm_dt = gm_data_x[1, 0] - gm_data_x[0, 0]
assert (gm_data_y[1, 0] - gm_data_y[0, 0]) == gm_dt
if gm_data_z is not None:
    assert (gm_data_z[1, 0] - gm_data_z[0, 0]) == gm_dt

# scale the ground motions
ag_x = gm_data_x[:, 1] * scaling
ag_y = gm_data_y[:, 1] * scaling
if gm_data_z is not None:
    ag_z = gm_data_z[:, 1] * scaling
else:
    ag_z = None

# ensure that the time-histories have the same number of data points
len_x = len(ag_x)
len_y = len(ag_y)
min_len = min(len_x, len_y)
if ag_z is not None:
    min_len = min(min_len, len(ag_z))
ag_x = ag_x[0:min_len]
ag_y = ag_y[0:min_len]
if ag_z is not None:
    ag_z = ag_z[0:min_len]


# some RSNs have an inappropriate duration (they are too short). We
# correct that here.
if rsn in (495,):
    # add 10 more seconds
    ag_x = np.array((*ag_x, *np.zeros(int(10.00 / gm_dt))))
    ag_y = np.array((*ag_y, *np.zeros(int(10.00 / gm_dt))))
    ag_z = np.array((*ag_z, *np.zeros(int(10.00 / gm_dt))))

if custom_path:
    output_folder = custom_path
else:
    output_folder = (
        f"results/response/{archetype}/{output_dir_name}/"
        f"{hazard_level}/gm{gm_number}"
    )
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# from osmg.graphics.preprocessing_3d import show
# show(mdl, loadcase, extrude=True)
# show(mdl, loadcase, extrude=False)

# #
# # modal analysis
# #

# modal_analysis = solver.ModalAnalysis(
#     mdl, {loadcase.name: loadcase}, num_modes=num_levels*3)
# modal_analysis.settings.store_forces = False
# modal_analysis.settings.store_fiber = False
# modal_analysis.settings.restrict_dof = [False]*6
# modal_analysis.run()


# for per in modal_analysis.results[loadcase.name].periods:
#     print(per)
# print(modal_analysis.results[loadcase.name].periods)

# from osmg.graphics.postprocessing_3d import show_deformed_shape
# show_deformed_shape(
#     modal_analysis, loadcase.name, 8, 0.00,
#     extrude=False, animation=True)

# mnstar = modal_analysis.modal_participation_factors(loadcase.name, 'x')[1]
# np.cumsum(mnstar)


#
# time-history analysis
#

t_bar = float(read_study_param(f"data/{archetype}/period"))

if damping == "rayleigh":
    damping_input = {
        "type": "rayleigh",
        "ratio": 0.02,
        "periods": [t_bar, t_bar / 10.00],
    }
elif damping == "modal":
    damping_input = {
        "type": "modal+stiffness",
        "num_modes": (num_levels) * 3,
        "ratio_modal": 0.02,
        "period": t_bar / 10.00,
        "ratio_stiffness": 0.001,
    }
else:
    raise ValueError(f"Invalid damping type: {damping}")


# define analysis object
nlth = solver.THAnalysis(mdl, {loadcase.name: loadcase})
nlth.settings.log_file = store_info(f"{output_folder}/log", gm_filenames)
if damping == "rayleigh":
    nlth.settings.solver = "Umfpack"
nlth.settings.store_fiber = False
nlth.settings.store_forces = False
nlth.settings.store_reactions = True
nlth.settings.store_release_force_defo = False
nlth.settings.specific_nodes = specific_nodes

# run the nlth analysis
nlth.run(
    analysis_dt,
    ag_x,
    ag_y,
    ag_z,
    gm_dt,
    damping=damping_input,
    print_progress=progress_bar,
    drift_check=0.10,  # 10% drift
    time_limit=47.95,  # hours
    dampen_out_residual=True,
    finish_time=finish_time,
)

# store response quantities

df = pd.DataFrame()
df["time--"] = np.array(nlth.time_vector)
df["Rtime--"] = np.array(nlth.results[loadcase.name].clock)
df["Rtime--"] -= df["Rtime--"].iloc[0]
df["Subdiv--"] = np.array(nlth.results[loadcase.name].subdivision_level)
for lvl in range(num_levels + 1):
    df[[f"FA-{lvl}-{j}" for j in range(1, 3)]] = nlth.retrieve_node_abs_acceleration(
        lvl_nodes[lvl], loadcase.name
    ).loc[:, "abs ax":"abs ay"]
    df[[f"FV-{lvl}-{j}" for j in range(1, 3)]] = nlth.retrieve_node_abs_velocity(
        lvl_nodes[lvl], loadcase.name
    ).loc[:, "abs vx":"abs vy"]
    if lvl > 0:
        us = nlth.retrieve_node_displacement(lvl_nodes[lvl], loadcase.name).loc[
            :, "ux":"uy"
        ]
        if lvl == 1:
            dr = us / level_heights[lvl - 1]
        else:
            us_prev = nlth.retrieve_node_displacement(
                lvl_nodes[lvl - 1], loadcase.name
            ).loc[:, "ux":"uy"]
            dr = (us - us_prev) / level_heights[lvl - 1]
        df[[f"ID-{lvl}-{j}" for j in range(1, 3)]] = dr

df["Vb-0-1"] = nlth.retrieve_base_shear(loadcase.name)[:, 0]
df["Vb-0-2"] = nlth.retrieve_base_shear(loadcase.name)[:, 1]

df.columns = pd.MultiIndex.from_tuples([x.split("-") for x in df.columns.to_list()])
df.sort_index(axis=1, inplace=True)

df.to_parquet(store_info(f"{output_folder}/results.parquet", gm_filenames))
