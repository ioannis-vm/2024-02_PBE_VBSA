"""
Generation of site hazard curves for ground motion selection for a
time-based assessment using OpenSHA PSHA output.

Note

  Assuming Poisson distributed earthquake occurence,


  p_occurence = 1 - exp(-t/T)

  where:
    P_exceedance is the probability of 1 or more occurences,
    t is the period of interest (1 year, 50 years, etc.),
    T is the return period (e.g. 475 years, 2475 years),
    1/T is called the `occurence rate` or `frequency`

"""

import os
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.util import read_study_param

# pylint: disable = invalid-name

np.seterr(divide="ignore")

out_path = "results/site_hazard"

if not os.path.exists(out_path):
    os.makedirs(out_path)

# Parse OpenSHA output
names = [
    "0p01",
    "0p02",
    "0p03",
    "0p05",
    "0p075",
    "0p1",
    "0p15",
    "0p2",
    "0p25",
    "0p3",
    "0p4",
    "0p5",
    "0p75",
    "1p0",
    "1p5",
    "2p0",
    "3p0",
    "4p0",
    "5p0",
    "7p5",
    "10p0",
]
periods = [
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
]
hzrd_crv_files = [f"{out_path}/{name}.txt" for name in names]

accelerations = []
MAPEs = []
MAFEs = []

dfs_sub = []

for filepath in hzrd_crv_files:
    with open(filepath, "r", encoding="utf-8") as f:
        line = f.read()
    contents = line.split("\n")
    i_begin = -1
    num_points = -1
    for i, c in enumerate(contents):
        words = c.split(" ")
        if words[0] == "X," and words[1] == "Y" and words[2] == "Data:":
            i_begin = i + 1
        if words[0] == "Num":
            num_points = int(words[2])
    i_end = i_begin + num_points
    data = np.genfromtxt(contents[i_begin:i_end])
    df = pd.DataFrame(data[:, 1], index=data[:, 0], columns=["MAPE"])
    df.index.name = "Sa"
    df["MAFE"] = -np.log(1.00 - df["MAPE"])
    dfs_sub.append(df)

df = pd.concat(dfs_sub, axis=1, keys=periods)
df.columns.names = ["T", "Type"]

df_mafe = df.xs("MAFE", axis=1, level=1)

# Plot hazard curves
plt.figure(figsize=(12, 10))
plt.grid(which="both")
for col in df_mafe.columns:
    plt.plot(df_mafe[col], "-s", label=f"T = {col:5.2f} s")
plt.xscale("log")
plt.yscale("log")
plt.axhline(0.00019999999999999985)
plt.axhline(0.04299432428103713)
plt.legend()
plt.xlabel("Earthquake Intensity $e$ [g]")
plt.ylabel("Mean annual frequency of exceedance $λ$")
plt.savefig("figures/hazard_curves.pdf")
# plt.show()
plt.close()

# save the hazard curves
df.to_csv(f"{out_path}/hazard_curves.csv")

# Obtain period-specific hazard curve
# Interpolate available hazard curves
t_bar = 1.00
target_mafes = []
for acc in df_mafe.index:
    vec = np.log(df_mafe.loc[acc, :])
    f = interp1d(periods, vec, kind="linear")
    target_mafes.append(np.exp(float(f(t_bar))))
df[("target", "MAFE")] = target_mafes


# Define interpolation functions for the period-specific hazard curve
# Interpolate: From MAFE λ to intensity e [g]
def fHazMAFEtoSa(mafe):
    """
    Interpolate Sa for a given MAFE
    """
    temp1 = interp1d(
        np.log(df[("target", "MAFE")]).to_numpy(),
        np.log(df.index.to_numpy()),
        kind="cubic",
    )
    return np.exp(temp1(np.log(mafe)))


# Interpolate: Inverse (From intensity e [g] to MAFE λ)
def fHazSatoMAFE(sa):
    """
    Interpolate MAFE for a given Sa
    """
    temp2 = interp1d(
        np.log(df.index).to_numpy(),
        np.log(df[("target", "MAFE")]).to_numpy(),
        kind="cubic",
    )
    return np.exp(temp2(np.log(sa)))


# Specify Intensity range
if t_bar <= 1.00:
    SaMin = 0.05
else:
    SaMin = 0.05 / t_bar
mafe_min = fHazSatoMAFE(SaMin)
SaMax = fHazMAFEtoSa(2e-4)
SaMin = fHazMAFEtoSa(0.50)
mafe_max = fHazSatoMAFE(SaMax)

# # plot target hazard curve
# import matplotlib.pyplot as plt
# plt.figure()
# plt.grid(which='both')
# for col in df_mafe.columns:
#     plt.plot(df_mafe[col],
#              label=f'T = {col:5.2f} s',
#              linewidth=0.2)
# plt.plot(
#     df[('target', 'MAFE')], color='red',
#     linewidth=3)
# plt.axvline(SaMin, color='k', linestyle='dashed')
# plt.axvline(SaMax, color='k', linestyle='dashed')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Earthquake Intensity $e$ [g]')
# plt.ylabel('Mean annual frequency of exceedance $λ$')
# plt.show()
# plt.close()


# Split intensity range to m intervals

m = int(read_study_param("data/study_vars/m"))

# Determine interval midpoints and endpoints

e_vec = np.linspace(SaMin, SaMax, m * 2 + 1)
mafe_vec = fHazSatoMAFE(e_vec)
mafe_des = 1.0 / 475.0
mafe_mce = 1.0 / 2475.0
e_des = fHazMAFEtoSa(mafe_des)
e_mce = fHazMAFEtoSa(mafe_mce)

# + this makes sure that two of the midpoints
#   will fall exactly on the design and MCE level
#   scenarios.

k = -1
if mafe_vec[-1] < mafe_des < mafe_vec[0]:
    # identify index closest to design lvl
    dif = np.full(m * 2 + 1, 0.00)
    for i, e in enumerate(e_vec):
        dif[i] = e_des - e
    k = 2 * np.argmin(dif[1::2] ** 2) + 1
    corr = np.full(len(e_vec), 0.00)
    corr[0 : k + 1] = np.linspace(0, dif[k], k + 1)
    corr[k::] = np.linspace(dif[k], 0, m * 2 - k + 1)
    e_vec = e_vec + corr
    mafe_vec = fHazSatoMAFE(e_vec)

if mafe_vec[-1] < mafe_mce < mafe_vec[0]:
    # identify index closest to MCE lvl
    dif = np.full(m * 2 + 1, 0.00)
    for i, e in enumerate(e_vec):
        dif[i] = e_mce - e
    k2 = 2 * np.argmin(dif[1::2] ** 2) + 1
    corr = np.full(len(e_vec), 0.00)
    corr[k + 1 : k2] = np.linspace(0, dif[k2], k2 - (k + 1))
    corr[k2::] = np.linspace(dif[k2], 0, m * 2 - k2 + 1)
    e_vec = e_vec + corr
    mafe_vec = fHazSatoMAFE(e_vec)


e_Endpoints = e_vec[::2]
MAFE_Endpoints = mafe_vec[::2]
e_Midpoints = e_vec[1::2]
MAFE_Midpoints = mafe_vec[1::2]
MAPE_Midpoints = 1 - np.exp(-MAFE_Midpoints)
return_period_midpoints = 1 / MAFE_Midpoints

delta_e = np.array([e_Endpoints[i + 1] - e_Endpoints[i] for i in range(m)])
delta_lamda = np.array([MAFE_Endpoints[i] - MAFE_Endpoints[i + 1] for i in range(m)])

fig, ax = plt.subplots()
ax.plot(df[("target", "MAFE")], "-", label="Hazard Curve", color="black")
ax.scatter(
    e_Endpoints,
    MAFE_Endpoints,
    s=80,
    facecolors="none",
    edgecolors="k",
    label="Interval Endpoints",
)
ax.scatter(
    e_Midpoints,
    MAFE_Midpoints,
    s=40,
    facecolors="k",
    edgecolors="k",
    label="Interval Midpoints",
)
for i, (x, y) in enumerate(zip(e_Midpoints, MAFE_Midpoints), 1):
    ax.text(x, y, str(i), ha="center", va="bottom")
ax.grid(which="both", linewidth=0.30)
ax.axvline(SaMin, color="k", linestyle="dashed")
ax.axvline(SaMax, color="k", linestyle="dashed")
ax.set(yscale="log")
ax.set(xlabel="$x$ [g]")
ax.set(ylabel=f"$λ(Sa(T={t_bar:.2f}) > x)$")
ax.set(xlim=(-0.40, 2.70))
ax.set(ylim=(0.50e-5, 5.0))
plt.savefig("figures/hazard_curve_points.pdf")
# plt.show()
plt.close()

# store hazard curve interval data
interv_df = pd.DataFrame(
    np.column_stack(
        (
            e_Midpoints,
            delta_e,
            delta_lamda,
            MAFE_Midpoints,
            MAPE_Midpoints,
            return_period_midpoints,
        )
    ),
    columns=["e", "de", "dl", "freq", "prob", "T"],
    index=range(1, m + 1),
)
interv_df.to_csv(f"{out_path}/Hazard_Curve_Interval_Data.csv")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Obtain Uniform Hazard Spectra for each midpoint #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

uhs_dfs = []

spectrum_idx = 0
for spectrum_idx in range(m):
    rs = []
    target_mafe = MAFE_Midpoints[spectrum_idx]
    for period in periods:
        log_mafe = np.log(df[(period, "MAFE")].to_numpy())
        log_sa = np.log(df.index.to_numpy())
        log_target_mafe = np.log(target_mafe)
        f = interp1d(log_mafe, log_sa, kind="cubic")
        log_target_sa = float(f(log_target_mafe))
        target_sa = np.exp(log_target_sa)
        rs.append(target_sa)
    uhs = np.column_stack((periods, rs))
    uhs_df = pd.DataFrame(uhs, columns=["T", "Sa"])
    uhs_df.set_index("T", inplace=True)
    uhs_dfs.append(uhs_df)

# write UHS data to files
for i, uhs_df in enumerate(uhs_dfs):
    uhs_df.to_csv(f"{out_path}/UHS_{i+1}.csv")

uhs_df = pd.concat(uhs_dfs, axis=1)
uhs_df.columns = [i + 1 for i in range(m)]

fig, ax = plt.subplots()
for col in uhs_df.columns:
    ax.plot(uhs_df[col], "k")
ax.grid(which="both")
ax.set(xscale="log", yscale="log")
ax.set(xlabel="Period [s]")
ax.set(ylabel="RotD50 Sa [g]")
ax.set(title="Uniform hazard spectra.")
plt.savefig("figures/uhss.pdf")
# plt.show()
plt.close()
