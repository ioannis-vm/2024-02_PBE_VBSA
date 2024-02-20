"""
Log all demand/capacity ratios for the braces
"""

import seaborn as sns
import matplotlib.pyplot as plt


archetype = ("scbf", "brbf")
story = ("3", "6", "9")
rcs = ("ii", "iv")

vals_brb_ii = []
vals_brb_iv = []
vals_cbf_ii = []
vals_cbf_iv = []

for ar in archetype:
    for st in story:
        for rc in rcs:
            path = f"results/design_logs/{ar}_{st}_{rc}.txt"
            with open(path, "r", encoding="utf-8") as f:
                contents = f.read()
            contents = contents.split("Brace Strength Check Capacity Ratios\n")[1]
            contents = contents.split("\n")[0]
            print(ar, st, rc, contents)
            contents = contents.replace("[", "").replace("]", "")
            contents = [float(x.replace("'", "")) for x in contents.split(", ")]
            if ar == "brbf":
                if rc == "ii":
                    vals_brb_ii.extend(contents)
                else:
                    vals_brb_iv.extend(contents)
            else:
                if rc == "ii":
                    vals_cbf_ii.extend(contents)
                else:
                    vals_cbf_iv.extend(contents)

fig, ax = plt.subplots()
sns.ecdfplot(vals_brb_ii, ax=ax, label="BRB RC II")
sns.ecdfplot(vals_brb_iv, ax=ax, label="BRB RC IV")
sns.ecdfplot(vals_cbf_ii, ax=ax, label="CBF RC II")
sns.ecdfplot(vals_cbf_iv, ax=ax, label="CBF RC IV")
ax.set(
    xlim=(0.00, 1.00),
    xlabel="D/C ratios",
    title="Empirical CDF plots of all brace D/C ratios\nseparated by system and RC.",
)
ax.legend()
plt.show()
