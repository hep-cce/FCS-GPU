"""
Utility functions for time, CSV parsing, JSON manipulation, and data processing.
"""

import re
import io
import csv

from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd


def get_current_time() -> str:
    """Return the current time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_timestamp() -> str:
    """Return the current timestamp as a formatted string."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def remove_trailing_commas(json_string: str) -> str:
    """Remove trailing commas from a JSON string."""
    json_string = re.sub(r",\s*([\]}])", r"\1", json_string)
    return json_string


def parse_csv_data(csv_data: str) -> Dict[str, str]:
    """Parse CSV data into a dictionary."""
    csv_file = io.StringIO(csv_data)
    reader = csv.DictReader(csv_file)
    dic = {}
    for row in reader:
        for key, value in row.items():
            dic[key] = value
    return dic


def make_launch_count_str(launch_count_list: List[Tuple[str, int]]) -> str:
    """Create a string summarizing launch counts."""

    def all_counts_equal(lst: List[Tuple[str, int]]) -> bool:
        return all(x[1] == lst[0][1] for x in lst)

    if all_counts_equal(launch_count_list):
        launch_count_str = f"n = {launch_count_list[0][1]} launches per kernel"
    else:
        print("Warning: Not all launch counts are equal")
        launch_count_str = "Launches per kernel: " + ",\n".join(
            [f"{image}: {count}" for image, count in launch_count_list]
        )
    return launch_count_str


def get_bar_list(patches: List) -> List[Tuple[float, float]]:
    """Return a sorted list of bar coordinates from patches."""
    bar_list = []
    kernel_patches = [patch for patch in patches if patch.get_height() != 0]
    for kernel_patch in kernel_patches:
        bar_xy = (
            (float(kernel_patch.get_x() + (kernel_patch.get_width() / 2))),
            float(kernel_patch.get_height()),
        )
        bar_list.append(bar_xy)
    bar_list.sort()
    return bar_list


def get_x_y_for_errorbar(axis) -> Tuple[List[float], List[float]]:
    """Return x and y coordinates for error bars from an axis."""
    bar_list = get_bar_list(axis.patches)
    return [x for x, y in bar_list], [y for x, y in bar_list]


def make_df_for_errorbar(df: pd.DataFrame, axis) -> pd.DataFrame:
    """Create a DataFrame with error bar coordinates."""
    bar_x, bar_y = get_x_y_for_errorbar(axis)

    unique_kernels = df["kernel"].unique()
    df["kernel"] = pd.Categorical(df["kernel"], categories=unique_kernels, ordered=True)
    sorted_df = df.sort_values(by=["kernel", "kernel_info"]).reset_index(drop=True)
    sorted_df["bar_x"] = bar_x
    sorted_df["bar_y"] = bar_y
    return sorted_df
