"""
Update scaling factors obtained from CS_Selection so that the suite
means perfectly match the UHSs at the conditioning period
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.util import read_study_param
from src.util import retrieve_peer_gm_spectra
from src.util import interpolate_pd_series

# load existing scaling factor dataframe
df_scaling = pd.read_csv(
    "results/site_hazard/required_records_and_scaling_factors.csv",
    index_col=[0, 1, 2],
)
df_scaling.columns = df_scaling.columns.astype(int)
df_scaling = df_scaling.sort_index(axis=0).sort_index(axis=1)

# archetype information
codes = ("smrf", "scbf", "brbf")
stories = ("3", "6", "9")
rcs = ("ii", "iv")
cases = [f"{c}_{s}_{r}" for c in codes for s in stories for r in rcs]

num_hz = int(read_study_param("data/study_vars/m"))

# mean_targets = []
# hz = 7
# # retrieve UHS
# uhs = pd.read_csv(
#     f'results/site_hazard/UHS_{hz+1}.csv',
#     index_col=0)['Sa']
# for arch in cases:
#     mean = pd.read_csv(
#         f'results/{arch}/site_hazard/target_mean_{hz+1}.csv',
#         index_col=0, header=None)[1]
#     mean_cs = np.exp(mean)
#     mean_targets.append(mean_cs)
# # plot suite and target
# fig, ax = plt.subplots()
# for trg in mean_targets:
#     ax.plot(trg*1.15, color='k', linestyle='dashed')
# ax.plot(uhs)
# ax.set(xscale='log', yscale='log')
# ax.grid(which='both', linewidth=0.10)
# ax.legend()
# plt.show()


with PdfPages("figures/gm_selection_spectra.pdf") as pdf:
    for arch in tqdm(cases):
        for hz in range(num_hz):
            t_bar = float(read_study_param(f"data/{arch}/period_closest"))

            # retrieve target mean from CS_Selection
            mean = pd.read_csv(
                f"results/site_hazard/{arch}/target_mean_{hz+1}.csv",
                index_col=0,
                header=None,
            )[1]
            mean_cs = np.exp(mean)

            # retrieve UHS
            uhs = pd.read_csv(f"results/site_hazard/UHS_{hz+1}.csv", index_col=0)[
                "Sa"
            ]

            stdv = pd.read_csv(
                f"results/site_hazard/{arch}/target_stdv_{hz+1}.csv",
                index_col=0,
                header=None,
            )[1]

            added_scaling = interpolate_pd_series(
                uhs, t_bar
            ) / interpolate_pd_series(mean_cs, t_bar)

            df_scaling.loc[(arch, f"hz_{hz+1}", "SF")] *= added_scaling

            # retrieve scaled ground motion records
            df_sub = df_scaling.loc[(arch, f"hz_{hz+1}")].T
            df_sub["RSN"] = df_sub["RSN"].astype(int)
            scaling = df_sub["SF"]
            rsns = df_sub["RSN"]
            scaling.index = rsns
            spectra = retrieve_peer_gm_spectra(rsns) * scaling

            suite_mean = np.log(spectra).mean(axis=1)
            suite_std = np.log(spectra).std(axis=1)

            # plot suite and target
            fig, ax = plt.subplots()
            figtitle = " ".join(arch.upper().split("_")) + ", " + f"HZ LVL {hz+1}"
            for i, col in enumerate(spectra):
                if i == 0:
                    lab = "Records"
                else:
                    lab = None
                ax.plot(spectra[col], linewidth=0.80, color="0.6", label=lab)
            ax.plot(np.exp(mean) * added_scaling, color="k", linestyle="dashed")
            ax.plot(
                np.exp(mean + 1.96 * stdv) * added_scaling,
                color="k",
                linestyle="dashed",
                label="Target $\\mu$ $\\pm$ 1.96 $\\sigma$",
            )
            ax.plot(
                np.exp(mean - 1.96 * stdv) * added_scaling,
                color="k",
                linestyle="dashed",
            )
            ax.plot(
                np.exp(suite_mean),
                color="red",
                linestyle="dotted",
                label="Suite $\\mu$ $\\pm$ 1.96 $\\sigma$",
            )
            ax.plot(
                np.exp(suite_mean + 1.96 * suite_std),
                color="red",
                linestyle="dotted",
            )
            ax.plot(
                np.exp(suite_mean - 1.96 * suite_std),
                color="red",
                linestyle="dotted",
            )
            ax.plot(uhs)
            ax.axvline(x=t_bar, color="k")
            ax.set(xscale="log", yscale="log", title=figtitle)
            ax.grid(which="both", linewidth=0.10)
            ax.legend()
            pdf.savefig()
            # plt.show()
            plt.close()

# store the updated scaling factors
df_scaling.to_csv(
    "results/site_hazard/required_records_and_scaling_factors_adjusted_to_cms.csv"
)
