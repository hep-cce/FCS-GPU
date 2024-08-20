"""
Functionality to plot HMA results:

- Load HMA results from a given result file
- Plot a comparison of multiple kernel run times with total elapsed run time and average run time
"""

import json
import os
from typing import Dict, Any, Union, Optional, List, Tuple
from datetime import datetime
from io import StringIO

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from hma_metrics import Metric
from util import make_launch_count_str, make_df_for_errorbar


def json_deserial(dct: Dict[str, Any]) -> Union[Dict[str, Any], Metric]:
    """JSON deserializer for custom objects"""

    if "data_type" in dct:
        if "float" in dct["data_type"]:
            return Metric(dct["name"], float, float(dct["value"]), dct["unit"])
        if "int" in dct["data_type"]:
            return Metric(dct["name"], int, int(dct["value"]), dct["unit"])
        if "str" in dct["data_type"]:
            return Metric(dct["name"], str, str(dct["value"]), dct["unit"])
        if "DataFrame" in dct["data_type"]:
            return Metric(
                dct["name"],
                pd.DataFrame,
                pd.read_json(StringIO(dct["value"])),
                dct["unit"],
            )
    if "timestamp" in dct:
        dct["timestamp"] = datetime.fromisoformat(dct["timestamp"])

    return dct


class HmaPlot:
    """Class for plotting HEPMiniApps results"""

    descriptive_names = {
        "sim_clean": "workspace_reset",
        "sim_A": "simulate",
        "sim_ct": "reduce",
        "sim_cp": "copy d->h",
    }

    def __init__(self, input_path: str) -> None:
        """Initialize HmaPlot with a file or directory"""
        self.input_path: str = input_path
        self.hma_data: List[Dict[str, Any]] = self.read_hma_input()
        self.kernels: List[pd.DataFrame]
        self.launch_count_list: List[Tuple[str, int]]
        self.kernels, self.launch_count_list = self.get_kernel_data()
        self.df_all: pd.DataFrame = self.prepare_dataframe()

    def read_hma_input(self) -> List[Dict[str, Any]]:
        """Read and deserialize input from a file or directory"""
        data: List[Dict[str, Any]] = []

        if os.path.isdir(self.input_path):
            input_files = [
                x for x in os.listdir(self.input_path) if x.endswith(".json")
            ]
            input_files.sort()
            input_files = [os.path.join(self.input_path, file) for file in input_files]
            for input_file in input_files:
                with open(input_file, "r", encoding="utf-8") as f:
                    result_list = json.load(f, object_hook=json_deserial)
                    data = result_list if not data else data + result_list
        elif os.path.isfile(self.input_path):
            with open(self.input_path, "r", encoding="utf-8") as f:
                data = json.load(f, object_hook=json_deserial)
        else:
            raise ValueError(
                f"Provided input path '{self.input_path}' is neither a valid file nor a directory."
            )

        return data

    def _prepare_kernel_info(self, kernel_run: Dict[str, Any]) -> str:
        """Prepare kernel information from the HMA data"""
        runner_label = kernel_run["info"]["runner_label"]
        image_info = kernel_run["info"]["image_info"]
        image_type = image_info["image_type"]
        image_tag = image_info["image_tag"]
        short_tag = image_tag.split("-")[0]
        kernel_info = runner_label + ":" + image_type + "_" + short_tag
        return kernel_info

    def get_kernel_data(self) -> Tuple[List[pd.DataFrame], List[Tuple[str, int]]]:
        """Extract kernel data and launch counts from the HMA data"""
        pr_models: List[pd.DataFrame] = []
        launch_count_list: List[Tuple[str, int]] = []
        kernel_runs = [
            hma_run for hma_run in self.hma_data if hma_run["metrics"] is not None
        ]

        for kernel_run in kernel_runs:
            metrics = kernel_run["metrics"]
            kernel_timing_idx = None
            launch_count_idx = None

            for m in metrics:
                if m.name == "kernel timing":
                    kernel_timing_idx = metrics.index(m)
                elif m.name == "launch count":
                    launch_count_idx = metrics.index(m)

            if kernel_timing_idx is not None:
                kernel_info = self._prepare_kernel_info(kernel_run)
                df_kernel = metrics[kernel_timing_idx].value
                df_kernel["kernel_info"] = kernel_info
                df_kernel["runner"] = kernel_run["info"]["runner_label"]
                df_kernel["image_type"] = kernel_run["info"]["image_info"]["image_type"]
                print(f"Kernel timing metrics found for {kernel_info}")
                pr_models.append(df_kernel)
            else:
                print(f"No kernel timing metrics found for {kernel_info}")

            if launch_count_idx is not None:
                launch_count = metrics[launch_count_idx].value
                launch_count_list.append((kernel_info, launch_count))
            else:
                print(f"No launch count found for {kernel_info}")

        return pr_models, launch_count_list

    def prepare_dataframe(self) -> pd.DataFrame:
        """Prepare a combined DataFrame with all kernel results"""
        kernel_df = pd.concat(self.kernels, ignore_index=True)
        kernel_df["kernel_info"] = kernel_df["kernel_info"].str.replace(
            r"^fcs-", "", regex=True
        )
        kernel_df["kernel"] = kernel_df["kernel"].replace(self.descriptive_names)
        return kernel_df

    def save_or_show_plot(
        self,
        save_plot: bool = True,
        filename: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> None:
        """Save or show the plot"""
        plt.tight_layout()

        if save_plot:
            if filename is None:
                plot_filename = (
                    f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                )
            elif os.path.isdir(filename):
                plot_filename = os.path.join(
                    filename,
                    f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
                )
            else:
                plot_filename = filename

            if group_by is not None:
                plot_filename = plot_filename.replace(".png", f"_{group_by}.png")

            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        else:
            plt.show()
        plt.close()

    def _make_total_run_time_plot(
        self, ax: plt.Axes, df: pd.DataFrame, title: str
    ) -> None:

        sns.barplot(
            ax=ax,
            x="kernel",
            y="total /s",
            hue="kernel_info",
            data=df,
            errorbar=None,
            capsize=0.1,
        )
        ax.set_title(title)
        ax.set_ylabel("Total elapsed run time [s]")
        ax.legend(title="Technology", loc="upper right", fontsize=12)

    def _make_avg_run_time_plot(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        launch_count_list: List[Tuple[str, int]],
        title: Optional[str] = None,
    ) -> None:

        sns.barplot(
            ax=ax,
            x="kernel",
            y="avg launch /us",
            hue="kernel_info",
            data=df,
            errorbar=None,
            capsize=0.1,
        )
        ax.set_title(title or "Average Kernel Run time in us")

        ax.set_ylabel("Average Run time and standard deviation [us]")
        ax.legend(title="Technology", loc="upper right", fontsize=12)

        props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
        launch_count_str = make_launch_count_str(launch_count_list)
        ax.text(
            0.02,
            0.95,
            launch_count_str,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=props,
        )

        df_errorbar = make_df_for_errorbar(df, ax)

        ax.errorbar(
            df_errorbar["bar_x"],
            df_errorbar["bar_y"],
            yerr=df_errorbar["std dev /us"],
            fmt="none",
            c="red",
            capsize=5,
        )
        ax.set_ylim(bottom=0)

    def plot(
        self,
        group_by: Optional[str] = None,
        save_plot: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """Plot the kernel run time comparisons, include error bars for average kernel run time"""
        if group_by == "runner":
            n = len(self.df_all["runner"].unique())
            _, axes = plt.subplots(2, n, figsize=((10 * n), 9))
            for i, runner in enumerate(self.df_all["runner"].unique()):
                df_runner = self.df_all[self.df_all["runner"] == runner]
                self._make_total_run_time_plot(
                    axes[0, i],
                    df_runner,
                    f"FastCaloSim Run Time comparison on {runner} for different technologies",
                )
                self._make_avg_run_time_plot(
                    axes[1, i],
                    df_runner,
                    [x for x in self.launch_count_list if runner in x[0]],
                    f"Average Run Time comparison: {runner} kernels",
                )
            filename = filename if filename is not None else "per_runner.png"
        elif group_by == "image_type":
            n = len(self.df_all["image_type"].unique())
            _, axes = plt.subplots(2, n, figsize=((10 * n), 9))
            for i, image_type in enumerate(self.df_all["image_type"].unique()):
                df_image_type = self.df_all[self.df_all["image_type"] == image_type]
                self._make_total_run_time_plot(
                    axes[0, i],
                    df_image_type,
                    f"FastCaloSim Run Time comparison of {image_type} across systems",
                )
                self._make_avg_run_time_plot(
                    axes[1, i],
                    df_image_type,
                    [x for x in self.launch_count_list if image_type in x[0]],
                    f"Average Run Time comparison: {image_type} kernels",
                )
            filename = filename if filename is not None else "per_image_type.png"
        else:
            _, axes = plt.subplots(2, 1, figsize=(10, 9))
            self._make_total_run_time_plot(
                axes[0],
                self.df_all,
                "FastCaloSim Run Time comparison for different technologies",
            )
            self._make_avg_run_time_plot(axes[1], self.df_all, self.launch_count_list)
            filename = filename if filename is not None else "all.png"

        self.save_or_show_plot(save_plot, filename, group_by)


def main(input_path, output):
    """Main function to create and save the plot from the HMA result file."""
    print(f"Plotting: input_path={input_path}, output={output}")
    plot = HmaPlot(input_path)
    plot.plot(
        save_plot=True,
        filename=output,
    )
    plot.plot(
        group_by="runner",
        save_plot=True,
        filename=output,
    )
    plot.plot(
        group_by="image_type",
        save_plot=True,
        filename=output,
    )
