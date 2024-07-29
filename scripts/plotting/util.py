import json
import pandas as pd
from datetime import datetime
from io import StringIO

import pprint

def construct_metric(name, type, value, unit):
    return {
        "name": name,
        "type": type,
        "value": value,
        "unit": unit if unit else None,
    }

def json_deserial(dct):
    """JSON deserializer for custom objects"""
    
    if "type" in dct:
        if "float" in dct["type"]:
            return construct_metric(
                dct["name"],
                float, 
                float(dct["value"]), 
                dct["unit"])
        elif "int" in dct["type"]:
            return construct_metric(
                dct["name"],
                int, 
                int(dct["value"]), 
                dct["unit"])
        elif "str" in dct["type"]:
            return construct_metric(
                dct["name"],
                str, 
                str(dct["value"]), 
                dct["unit"])
        elif "DataFrame" in dct["type"]:
            return construct_metric(
                dct["name"],
                pd.DataFrame, 
                pd.read_json(StringIO(dct["value"])), 
                dct["unit"])
    elif "timestamp" in dct:
        dct["timestamp"] = datetime.fromisoformat(dct["timestamp"])

    return dct

def find_list_index(metrics, name):
    #return first index of the list that matches the name
    indices = [index for index, metric in enumerate(metrics) if metric["name"] == name]
    if indices:
        return indices[0]
    return None

def parse_json_files(files):
    pr_models = []
    launch_count_list = []

    for file in files:

        with open(file, 'r') as file:
            data = json.load(file, object_hook=json_deserial)
            image = data["info"]["image_type"] + "_" + data["info"]["image_tag"]
            technology = data["info"]["image_type"]

            if data["metrics"]:
                metrics = data["metrics"]
                kernel_timing_idx = find_list_index(metrics, "kernel timing")
                launch_count_idx = find_list_index(metrics, "launch count")

                if kernel_timing_idx != None:
                    df_metrics = metrics[kernel_timing_idx]["value"]
                    df_metrics["technology"] = technology
                    df_metrics["image"] = image
                    pr_models.append(df_metrics)
                else:
                    print(f"No kernel timing metrics found for {image}")

                if launch_count_idx != None:
                    launch_count = metrics[launch_count_idx]["value"]
                    launch_count_list.append((image, launch_count))
                    print(f"Launch count appended for {image}: {launch_count}")
                else:
                    print(f"No launch count found for {image}")

            else:
                print(f"No metrics found for {image}")

    df_all = pd.concat(pr_models, ignore_index=True)
    df_all['technology'] = df_all['technology'].str.replace(r'^fcs-', '', regex=True)
    return (df_all, launch_count_list)

def make_launch_count_str(launch_count_list):
    # Check if every count in launch_count_list: [(image, count), ...] is the same
    def all_counts_equal(lst):
        return all(x[1] == lst[0][1] for x in lst)

    if all_counts_equal(launch_count_list):
        launch_count_str = f'n = {launch_count_list[0][1]} launches per kernel'
    else:
        print('Warning: Not all launch counts are equal')
        launch_count_str = f'Launches per kernel: {",\n".join([f"{image}: {count}" for image, count in launch_count_list])}'
    return launch_count_str

def get_bar_list(patches):
    bar_list = []
    kernel_patches = [patch for patch in patches if patch.get_height() != 0]
    for kernel_patch in kernel_patches:
        bar = ( 
            (float(kernel_patch.get_x() + (kernel_patch.get_width() / 2))),
            float(kernel_patch.get_height())
            )
        bar_list.append(bar)
    bar_list.sort()
    return bar_list

def get_x_y_for_errorbar(axis):
    bar_list = get_bar_list(axis.patches)
    return [x for x, y in bar_list], [y for x, y in bar_list]

def make_df_for_errorbar(df, axis):
    bar_x, bar_y = get_x_y_for_errorbar(axis)

    # Sort the DataFrame by 'kernel' using Categorical dtype to maintain the current order, and then by 'technology'
    unique_kernels = df['kernel'].unique()
    df['kernel'] = pd.Categorical(df['kernel'], categories=unique_kernels, ordered=True)
    sorted_df = df.sort_values(by=['kernel', 'technology']).reset_index(drop=True)
    sorted_df['bar_x'] = bar_x
    sorted_df['bar_y'] = bar_y
    return sorted_df