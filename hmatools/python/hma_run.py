"""
Module for handling FastCaloSim run results.

This module defines the HmaRun class, which encapsulates the information and metrics
from a FastCaloSim run log file. It provides methods to initialize the run data,
convert it to a dictionary, and access the run info and metrics.
"""

import os
from typing import Union, Dict, List, Optional
from collections.abc import Iterable

from run_info import RunInfo
from hma_metrics import HmaMetrics


class HmaRun:
    """Representation of one HmaRun, including relevant info and metrics"""

    def __init__(self, logfile: Union[str, os.PathLike]) -> None:
        """Initialize FastCaloSim result object with the path to the input file."""
        if not os.path.isfile(logfile):
            raise ValueError(f"The provided input path '{logfile}' is not a file.")
        self.logfile = logfile
        with open(logfile, "r", encoding="utf-8") as f:
            logfile_content = f.read()
            info_data, kernel_data = logfile_content.split("- Setup")

        self.info = RunInfo(logfile, info_data)
        self.metrics = HmaMetrics(logfile, kernel_data)

    def to_dict(
        self,
    ) -> Dict[
        str, Union[Dict[str, str], Optional[List[Dict[str, Union[str, int, float]]]]]
    ]:
        """Convert the FastCaloSim result object to a dictionary."""
        if isinstance(self.metrics.metrics, Iterable):
            metrics = (
                [m.__dict__ for m in self.metrics.metrics]
                if self.metrics.metrics
                else None
            )
            return {
                "info": self.info.__dict__,
                "metrics": metrics,
            }
        print(f"No metrics found for {self.logfile}")
        return {"info": self.info.__dict__, "metrics": None}

    def get_info(self) -> RunInfo:
        """Public method to get the RunInfo object."""
        return self.info

    def get_metrics(self) -> HmaMetrics:
        """Public method to get the HmaMetrics object."""
        return self.metrics
