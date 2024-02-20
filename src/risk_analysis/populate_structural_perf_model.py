"""
Semi-automate the population of structural components
"""

import json
import pandas as pd


def get_hss_weight(sec):
    """
    Retrieve the weight of an HSS from section data.
    """
    with open("data/sections.json", "rb") as f:
        secs = json.load(f)
    return secs[sec]["W"]


def get_cmp_description(cmp_id):
    """
    Get the description of a component based on its cmp_id.
    """
    dmg_df = pd.read_csv(
        "data/performance/input_damage.csv", header=[0, 1], index_col=0
    )
    return dmg_df.loc[cmp_id, "Description"].unique()[0]


def append_row(
    dataframe,
    index,
    units,
    location,
    direction,
    theta_0,
    theta_1,
    family,
    blocks,
    inactive,
    description,
):
    """
    Appends a row to the dataframe
    """
    new_dataframe = pd.DataFrame(
        [
            [
                units,
                location,
                direction,
                theta_0,
                theta_1,
                family,
                blocks,
                inactive,
                description,
            ]
        ],
        columns=dataframe.columns,
        index=[index],
    )
    return pd.concat((dataframe, new_dataframe))


def get_sections(archetype, description):
    """
    Get the sections from the design logs
    """
    with open(f"results/design_logs/{archetype}.txt", "r", encoding="utf-8") as f:
        contents = f.read()
    split = contents.split(description + "\n")
    if len(split) == 2:
        seclist = (
            split[1].split("]")[0].replace("[", "").replace("'", "").replace('"', "")
        ).split(", ")
        return seclist
    if len(split) == 3:
        seclist_i = (
            split[1].split("]")[0].replace("[", "").replace("'", "").replace('"', "")
        ).split(", ")
        seclist_j = (
            split[2].split("]")[0].replace("[", "").replace("'", "").replace('"', "")
        ).split(", ")
        return (seclist_i, seclist_j)
    return None


def define_perf_model(archetype):
    """
    Define the structural performance model for a given archetype
    designation
    """
    system, levels, rc = archetype.split("_")

    # initialize
    header = [
        "Units",
        "Location",
        "Direction",
        "Theta_0",
        "Theta_1",
        "Family",
        "Blocks",
        "Inactive",
        "Description",
    ]
    perf_model = pd.DataFrame(columns=header)

    # COMPONENT: Bolted shear tab gravity connections
    if system == "smrf":
        perf_model = append_row(
            perf_model,
            "B.10.31.001",
            "ea",
            "all",
            "1",
            "44",
            None,
            None,
            "",
            None,
            get_cmp_description("B.10.31.001"),
        )
        perf_model = append_row(
            perf_model,
            "B.10.31.001",
            "ea",
            "all",
            "2",
            "40",
            None,
            None,
            "",
            None,
            get_cmp_description("B.10.31.001"),
        )
    elif system == "scbf":
        pass
    elif system == "brbf":
        pass

    # COMPONENT: Steel Column Base Plates
    section_lists = []
    section_counts = []

    if system == "smrf":
        # exterior columns, outer frame
        if rc == "ii":
            sec = get_sections(archetype, "Exterior Columns")[0]
            section_lists.append(sec)
        else:
            sec = get_sections(archetype, "Exterior Columns")[0][0]
            section_lists.append(sec)
        section_counts.append("4")

        # interior columns, outer frame
        if rc == "ii":
            sec = get_sections(archetype, "Interior Columns")[0]
            section_lists.append(sec)
        else:
            sec = get_sections(archetype, "Interior Columns")[0][0]
            section_lists.append(sec)
        section_counts.append("6")

        if rc == "iv":
            # if RC IV, exterior columns, inner frame
            sec = get_sections(archetype, "Exterior Columns")[1][0]
            section_lists.append(sec)
            section_counts.append("4")
            # if RC IV, interior columns, inner frame
            sec = get_sections(archetype, "Interior Columns")[1][0]
            section_lists.append(sec)
            section_counts.append("2")

    else:
        # system is either SCBF or BRBF
        sec = get_sections(archetype, "Columns")[0]
        section_lists.append(sec)
        if rc == "ii":
            section_counts.append("6")
        else:
            section_counts.append("12")

    for seclist, count in zip(section_lists, section_counts):
        weight = float(seclist.split("X")[1])
        if weight < 150:
            cmpid = "B.10.31.011a"
        elif weight < 300:
            cmpid = "B.10.31.011b"
        else:
            cmpid = "B.10.31.011c"
        perf_model = append_row(
            perf_model,
            cmpid,
            "ea",
            "1",
            "1,2",
            count,
            None,
            None,
            "",
            None,
            get_cmp_description(cmpid),
        )

    # COMPONENT: Welded column splices
    section_lists = []
    section_counts = []

    if system == "smrf":
        # exterior columns, outer frame
        if rc == "ii":
            sec = get_sections(archetype, "Exterior Columns")
            section_lists.append(sec)
        else:
            sec = get_sections(archetype, "Exterior Columns")[0]
            section_lists.append(sec)
        section_counts.append("4")

        # interior columns, outer frame
        if rc == "ii":
            sec = get_sections(archetype, "Interior Columns")
            section_lists.append(sec)
        else:
            sec = get_sections(archetype, "Interior Columns")[0]
            section_lists.append(sec)
        section_counts.append("6")

        if rc == "iv":
            # if RC IV, exterior columns, inner frame
            sec = get_sections(archetype, "Exterior Columns")[1]
            section_lists.append(sec)
            section_counts.append("4")
            # if RC IV, interior columns, inner frame
            sec = get_sections(archetype, "Interior Columns")[1]
            section_lists.append(sec)
            section_counts.append("2")

    else:
        # system is either SCBF or BRBF
        sec = get_sections(archetype, "Columns")
        section_lists.append(sec)
        if rc == "ii":
            section_counts.append("6")
        else:
            section_counts.append("12")

    for seclist, count in zip(section_lists, section_counts):
        splices = []
        for istory in range(1, int(levels)):
            # add a splice if the section has changed
            if seclist[istory] != seclist[istory - 1]:
                # add a splice here
                sec = seclist[istory - 1]
                weight = float(sec.split("X")[1])
                if weight < 150:
                    cmpid = "B.10.31.021a"
                elif weight < 300:
                    cmpid = "B.10.31.021b"
                else:
                    cmpid = "B.10.31.021c"
                perf_model = append_row(
                    perf_model,
                    cmpid,
                    "ea",
                    f"{istory+1}",
                    "1,2",
                    count,
                    None,
                    None,
                    "",
                    None,
                    get_cmp_description(cmpid),
                )
                splices.append(istory)

            # also add a splice if the same section is used for three
            # consecutive stories
            if istory >= 2:
                if (
                    (seclist[istory] == seclist[istory - 1])
                    and (seclist[istory] == seclist[istory - 2])
                    and (istory not in splices)
                    and (istory - 1 not in splices)
                    and (istory - 2 not in splices)
                ):
                    # add a splice here
                    sec = seclist[istory - 1]
                    weight = float(sec.split("X")[1])
                    if weight < 150:
                        cmpid = "B.10.31.021a"
                    elif weight < 300:
                        cmpid = "B.10.31.021b"
                    else:
                        cmpid = "B.10.31.021c"
                    perf_model = append_row(
                        perf_model,
                        cmpid,
                        "ea",
                        f"{istory+1}",
                        "1,2",
                        count,
                        None,
                        None,
                        "",
                        None,
                        get_cmp_description(cmpid),
                    )
                    splices.append(istory)
                    splices.append(istory)

    # COMPONENT: steel moment connection, other than RBS

    if system == "smrf":
        # beam on one side

        section_lists = []
        section_counts = []

        if rc == "ii":
            section_lists.append(get_sections(archetype, "Beams"))
            section_counts.append(4)
        else:
            section_lists.append(get_sections(archetype, "Beams")[0])
            section_counts.append(4)
            section_lists.append(get_sections(archetype, "Beams")[1])
            section_counts.append(4)

        for seclist, count in zip(section_lists, section_counts):
            for istory in range(int(levels)):
                sec = seclist[istory]
                depth = float(sec.split("X")[0].replace("W", ""))
                if depth <= 27.0:
                    cmpid = "B.10.35.021"
                else:
                    cmpid = "B.10.35.022"
                perf_model = append_row(
                    perf_model,
                    cmpid,
                    "ea",
                    f"{istory+1}",
                    "1,2",
                    count,
                    None,
                    None,
                    "",
                    None,
                    get_cmp_description(cmpid),
                )

        # beam on both sides

        section_lists = []
        section_counts = []

        if rc == "ii":
            section_lists.append(get_sections(archetype, "Beams"))
            section_counts.append(6)
        else:
            section_lists.append(get_sections(archetype, "Beams")[0])
            section_counts.append(6)
            section_lists.append(get_sections(archetype, "Beams")[1])
            section_counts.append(2)

        for seclist, count in zip(section_lists, section_counts):
            for istory in range(int(levels)):
                sec = seclist[istory]
                depth = float(sec.split("X")[0].replace("W", ""))
                if depth <= 27.0:
                    cmpid = "B.10.35.031"
                else:
                    cmpid = "B.10.35.032"
                perf_model = append_row(
                    perf_model,
                    cmpid,
                    "ea",
                    f"{istory+1}",
                    "1,2",
                    count,
                    None,
                    None,
                    "",
                    None,
                    get_cmp_description(cmpid),
                )

    # COMPONENT: SCBF braces

    if system == "scbf":
        section_lists = []
        section_counts = []

        section_lists.append(get_sections(archetype, "Braces"))
        if rc == "ii":
            section_counts.append(4)
        else:
            section_counts.append(8)

        for seclist, count in zip(section_lists, section_counts):
            for istory in range(int(levels)):
                sec = seclist[istory]
                weight = get_hss_weight(sec)
                if weight <= 40.0:
                    cmpid = "B.10.33.002a"
                elif weight <= 100.00:
                    cmpid = "B.10.33.002b"
                else:
                    cmpid = "B.10.33.002c"

                perf_model = append_row(
                    perf_model,
                    cmpid,
                    "ea",
                    f"{istory+1}",
                    "1,2",
                    count,
                    None,
                    None,
                    "",
                    None,
                    get_cmp_description(cmpid),
                )

    # COMPONENT: BRBF braces

    if system == "brbf":
        section_lists = []
        section_counts = []

        section_lists.append(get_sections(archetype, "BRBs"))
        if rc == "ii":
            section_counts.append(4)
        else:
            section_counts.append(8)

        for seclist, count in zip(section_lists, section_counts):
            for istory in range(int(levels)):
                sec = seclist[istory]
                core_area = float(sec)
                if core_area <= 7.00:
                    cmpid = "B.10.33.111a"
                else:
                    cmpid = "B.10.33.111b"

                perf_model = append_row(
                    perf_model,
                    cmpid,
                    "ea",
                    f"{istory+1}",
                    "1,2",
                    count,
                    None,
                    None,
                    "",
                    None,
                    get_cmp_description(cmpid),
                )

    perf_model.sort_index(inplace=True)

    perf_model.to_csv(f"data/{archetype}/performance/input_cmp_quant.csv")


cases = []
for system in ("smrf", "scbf", "brbf"):
    for levels in ("3", "6", "9"):
        for rc in ("ii", "iv"):
            archetype = f"{system}_{levels}_{rc}"
            cases.append(archetype)

for item in cases:
    define_perf_model(item)
