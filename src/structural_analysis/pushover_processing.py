"""
Processing of pushover analysis results

"""

import os
import dask.bag as db
from dask.distributed import Client
import pandas as pd
import matplotlib.pyplot as plt


def process_item(item):
    archetype = item[0]
    direction = item[1]

    data_dir = f"results/response/{archetype}/pushover"
    df = pd.read_csv(f"{data_dir}/results_{direction}.csv", index_col=0)
    df.set_index("Vb", inplace=True)

    if not os.path.exists("figures/pushover"):
        os.makedirs("figures/pushover")

    if ("scbf" in archetype) or ("brbf" in archetype):
        if "iv" in archetype:
            scaling = 4.00
        else:
            scaling = 2.00
    else:
        scaling = 2.00

    if "3" in archetype:
        xmax = 30.00
        if "ii" in archetype:
            ymax = 4000.00
        else:
            ymax = 10000.00
    if "6" in archetype:
        xmax = 40.00
        if "ii" in archetype:
            ymax = 5000.00
        else:
            ymax = 12000.00
    if "9" in archetype:
        xmax = 60.00
        if "ii" in archetype:
            ymax = 6000.00
        else:
            ymax = 16000.00

    # # determine design base shear
    # with open(f'results/design_logs/{archetype}.txt') as f:
    #     contents = f.read()
    # vbd = float(contents.split('V_b_elf = ')[1].split(' kips')[0])

    # if 'smrf' in archetype:
    #     omega = 5.50
    # if 'scbf' in archetype:
    #     omega = 5.00
    # else:
    #     omega = 5.00

    fig, ax = plt.subplots(figsize=(5, 2.2))
    for col in df.columns:
        ax.plot(df[col], df[col].index / 1e3 * scaling, color="k")
    vbmax = max(df.index / 1e3 * scaling)
    ax.text(
        0.95,
        0.05,
        f"Vb max = {vbmax:.0f} kips",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.set(xlim=(0.00, xmax * 1.02), ylim=(0.00, ymax * 1.02))
    ax.grid(which="both")
    ax.set(
        xlabel="Displacements [in]",
        ylabel="Base Shear [kips]",
        title=f"{archetype.upper().replace('_', ' ')}",
    )
    # ax.axhline(y=vbd, color='black')
    # ax.axhline(y=vbd*omega, color='black')
    # fig.show()
    fig.tight_layout()
    fig.savefig(f"figures/pushover/{archetype}_{direction}.pdf")
    plt.close()


# Create a Dask client with a specific number of workers/cores
client = Client(n_workers=12)  # Set the desired number of workers/cores here

# Create a list of items to process
items = []
for cd in ("smrf", "scbf", "brbf"):
    for st in ("3", "6", "9"):
        for rc in ("ii", "iv"):
            for dr in ("x", "y"):
                items.append([f"{cd}_{st}_{rc}", dr])

# Convert the list to a Dask bag
bag = db.from_sequence(items)

# Apply the function in parallel using Dask's map operation
results = bag.map(process_item)

# Compute the results and retrieve the output
output = results.compute()
