"""
Utility functions
"""

import os
from datetime import datetime
import socket
import sys
import hashlib
import shutil
from importlib.metadata import distributions
from io import StringIO
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import git


def read_study_param(param_path):
    """
    Read a study parameter from a file.
    """
    with open(param_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def retrieve_peer_gm_data(rsn, out_type="filenames", uhs=False):
    """
    Parses the `_SearchResults.csv` file inside a ground motion group
    folder and retrieves the unscaled RotD50 response spectrum or the
    ground motion filenames.
    """

    if not uhs:

        # determine record group
        groups_df = pd.read_csv(
            "results/site_hazard/ground_motion_group.csv", index_col=0
        )
        groups_df.index = groups_df.index.astype(int)

        if rsn not in groups_df.index:
            raise ValueError(f"rsn not found in round_motion_group.csv: {rsn}")

        group = groups_df.at[rsn, "group"]

        rootdir = f"data/ground_motions/PEERNGARecords_Unscaled({group})"

    else:

        rootdir = "data/ground_motions/uhs"

    file_path = f"{rootdir}/_SearchResults.csv"

    with open(file_path, "r", encoding="utf-8") as f:
        contents = f.read()

    if out_type == "filenames":
        contents = contents.split(" -- Summary of Metadata of Selected Records --")[
            1
        ].split("\n\n")[0]
        data = StringIO(contents)

        df = pd.read_csv(data, index_col=2)

        if rsn not in df.index:
            raise ValueError(f"rsn not found: {rsn}")

        filenames = df.loc[
            rsn,
            [
                " Horizontal-1 Acc. Filename",
                " Horizontal-2 Acc. Filename",
                " Vertical Acc. Filename",
            ],
        ].to_list()

        result = []
        for filename in filenames:
            if "---" in filename:
                result.append(None)
            else:
                result.append(f"{rootdir}/" + filename.strip())

        return result

    if out_type == "spectrum":
        contents = contents.split(" -- Scaled Spectra used in Search & Scaling --")[
            1
        ].split("\n\n")[0]
        data = StringIO(contents)

        df = pd.read_csv(data, index_col=0)
        # drop stats columns
        df = df.drop(
            columns=[
                "Arithmetic Mean pSa (g)",
                "Arithmetic Mean + Sigma pSa (g)",
                "Arithmetic Mean - Sigma pSa (g)",
            ]
        )
        df.columns = [x.split(" ")[0].split("-")[1] for x in df.columns]
        df.columns.name = "RSN"
        df.columns = df.columns.astype(int)
        df.index.name = "T"

        if rsn not in df.columns:
            raise ValueError(f"rsn not found: {rsn}")

        return df[rsn]

    raise ValueError("Unsupported out_type: {out_type}")


def retrieve_peer_gm_spectra(rsns):
    """
    Uses retrieve_peer_gm_data to prepare a dataframe with response
    spectra for the given RSNs
    """

    rsn_dfs = []
    for rsn in rsns:
        rsn_df = retrieve_peer_gm_data(rsn, out_type="spectrum")
        rsn_dfs.append(rsn_df)
    df = pd.concat(rsn_dfs, keys=rsns, axis=1)

    return df


def interpolate_pd_series(series, values):
    """
    Interpolates a pandas series for specified index values.
    """
    idx_vec = series.index.to_numpy()
    vals_vec = series.to_numpy()
    ifun = interp1d(idx_vec, vals_vec)
    if isinstance(values, float):
        return float(ifun(values))
    if isinstance(values, np.ndarray):
        return ifun(values)
    return ValueError(f"Invalid datatype: {type(values)}")


def check_last_line(file_path, target_string):
    """
    Checks if the last line of a file contains a specific string.

    Args:
        file_path (str): The path to the file.
        target_string (str): The string to search for in the last line.

    Returns:
        bool: True if the last line contains the target string, False otherwise.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Check if the file is not empty
    if lines:
        last_line = lines[-1].strip()  # Remove leading/trailing whitespace

        # Check if the last line contains the target string
        if target_string in last_line:
            return True

    return False


def check_any_line(file_path, target_string):
    """
    Checks if any line of a file contains a specific string.

    Args:
        file_path (str): The path to the file.
        target_string (str): The string to search for in the last line.

    Returns:
        bool: True if the last line contains the target string, False otherwise.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        all_contents = file.read()

    # Check if the file is not empty
    if all_contents:
        if target_string in all_contents:
            return True

    return False


def get_any_line(file_path, target_string):
    """
    Checks if any line of a file contains a specific string.
    If it does, it returns that line.

    Args:
        file_path (str): The path to the file.
        target_string (str): The string to search for in the last line.

    Returns:
        str: The line
    """
    with open(file_path, "r", encoding="utf-8") as file:
        all_contents = file.readlines()

    # Check if the file is not empty
    if all_contents:
        for line in all_contents:
            if target_string in line:
                return line

    return None


def check_logs(path):
    """
    Check the logs of a nonlinear analysis
    """

    exists = os.path.exists(path) and os.path.isfile(path)
    if not exists:
        return "does not exist"
    inter = check_any_line(path, "Analysis interrupted")
    if inter:
        return "interrupted"
    fail = check_any_line(path, "Analysis failed to converge")
    if fail:
        return "failed"
    return "finished"


def calculate_input_file_info(file_list):
    """
    Calculate a SHA256 checksum for each file in the file_list.
    Returns a descriptive string including file name,
    checksum, last modified date, and filesize.
    """
    file_info_strings = []

    for file_name in file_list:

        # Calculate individual file SHA256 checksum
        hash_sha256 = hashlib.sha256()
        # pylint: disable=cell-var-from-loop
        with open(file_name, "rb") as file:
            for byte_block in iter(lambda: file.read(4096), b""):
                hash_sha256.update(byte_block)

        file_checksum = hash_sha256.hexdigest()
        # Get last modified time and size of the file
        file_stats = os.stat(file_name)
        last_modified_date = datetime.fromtimestamp(file_stats.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        file_size = file_stats.st_size

        # Convert size to a human-friendly format
        suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
        human_size = file_size
        i = 0
        while human_size >= 1024 and i < len(suffixes) - 1:
            human_size /= 1024.0
            i += 1
        file_size_human = f"{human_size:.2f} {suffixes[i]}"

        # Combine information into a string for each file
        file_info = (
            f"    File: {os.path.basename(file_name)}\n"
            f"    Checksum: {file_checksum}\n"
            f"    Last Modified: {last_modified_date}\n"
            f"    Size: {file_size_human}\n"
        )
        file_info_strings.append(file_info)

    return "\n".join(file_info_strings)


def store_info(path, input_data_paths=None, seeds=None):
    """
    Store metadata enabling reproducibility of results
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_content = f"Time: {timestamp}\n"

    # Check if the file already exists
    if os.path.isfile(path):
        timestamp_for_backup = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = os.path.join(
            os.path.dirname(path), "replaced_on_" + timestamp_for_backup
        )
        os.makedirs(backup_folder, exist_ok=True)
        shutil.move(path, backup_folder)
        metadata_content += f"Moved existing {path} in {backup_folder}\n"
        info_path = path + ".info"
        if os.path.isfile(info_path):
            shutil.move(info_path, backup_folder)
            metadata_content += f"Moved existing {info_path} in {backup_folder}\n"

    try:
        # Get the current repo SHA
        sha = git.Repo(os.getcwd()).head.commit.hexsha
        metadata_content += f"Repo SHA: {sha}\n"
    except git.exc.InvalidGitRepositoryError:
        metadata_content += "Repo SHA: Git repo not found.\n"

    metadata_content += f"Hostname: {socket.gethostname()}\n"
    metadata_content += f"Python version: {sys.version}\n"

    # Get a list of installed packages and their versions using importlib.metadata
    installed_packages = [
        f"    {distribution.metadata['Name']}=={distribution.version}"
        for distribution in distributions()
    ]
    installed_packages_str = "\n".join(installed_packages)
    metadata_content += f"Installed packages:\n{installed_packages_str}\n"

    command_line_args = " ".join(sys.argv)
    metadata_content += f"Command line arguments: {command_line_args}\n"

    if input_data_paths:
        file_info = calculate_input_file_info(input_data_paths)
        metadata_content += "Input file information:\n"
        metadata_content += file_info

    if seeds:
        metadata_content += f"Random Seeds: {seeds}\n"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path + ".info", "w", encoding="utf-8") as file:
        file.write(metadata_content)

    return path
