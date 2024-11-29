"""
This module provides classes for parsing and storing metrics from log files.

Classes:
    Metric: A data class representing a single metric.
    HmaMetrics: A class for parsing and storing metrics from a log file.
"""

import re
import io
import os

from dataclasses import dataclass
from typing import Union, List, Optional

import pandas as pd


@dataclass
class Metric:
    """
    Represents a single metric with a name, data type, value, and optional unit.
    """

    name: str
    data_type: type
    value: Union[float, int, str, pd.DataFrame]
    unit: Optional[str] = None


class HmaMetrics:
    """
    Parses and stores metrics from a log file.
    """

    def __init__(self, logfile: Union[str, os.PathLike], data: str) -> None:
        self.logfile = logfile
        self.metrics = self._parse_metrics(data)

    def _parse_metrics(self, data: str) -> List[Metric]:
        kernel_lines = []
        capture_kernel_timing = False

        for line in data.splitlines():
            if not capture_kernel_timing and "Event: 9750" in line:
                capture_kernel_timing = True
                continue

            if capture_kernel_timing:
                if "exiting early" in line:
                    break
                kernel_lines.append(line.strip())

        if kernel_lines:
            return self._parse_kernel_data(kernel_lines)
        print(f"No kernel information found for {self.logfile}")
        return []

    def _parse_kernel_lines(self, kernel_lines: List[str]) -> List[Metric]:
        metrics = []
        kernel_timing_lines = []
        for line in kernel_lines:
            if ":" in line:
                m = self._process_kernel_line(line)
                if m is not None:
                    metrics.append(m)
            elif line.split()[0] in [
                "kernel",
                "sim_clean",
                "sim_A",
                "sim_ct",
                "sim_cp",
            ]:
                elements = line.strip()
                kernel_timing_lines.append(elements)
            elif "Time for Chain" in line:
                chain_time_match = re.match(
                    r"Time for Chain (\d+) is ([0-9.]+) (\w+)", line.strip()
                )
                if chain_time_match:
                    chain_id, value_num, unit = chain_time_match.groups()
                    m = Metric(
                        f"Time for Chain {chain_id}", float, float(value_num), unit
                    )
                    metrics.append(m)
            elif "launch count" in line:
                match = re.search(r"launch count\s+(\d+)\s*(\+\d+)?", line)
                if match:
                    launch_count = int(match.group(1))
                    m = Metric("launch count", str, str(launch_count), None)
                    metrics.append(m)

        if kernel_timing_lines:
            name = "kernel timing"
            m = Metric(
                name,
                pd.DataFrame,
                self._kernel_timing_df(kernel_timing_lines[1:]),
                None,
            )
            metrics.append(m)
        return metrics

    def _parse_kernel_data(self, kernel_lines: List[str]) -> List[Metric]:
        metrics = self._parse_kernel_lines(kernel_lines)
        if metrics:
            print(f"Found {len(metrics)} metrics for {self.logfile}")
            return metrics
        print(f"No kernel timing information found for {self.logfile}")
        return []

    def _process_kernel_line(self, line: str) -> Optional[Metric]:
        if "GPU memory used(MB)" in line:
            special_case_match = re.match(r"(.+?)\((\w+)\):\s*([0-9.]+)", line.strip())
            if special_case_match:
                name, unit, value = special_case_match.groups()
                return Metric(name, int, int(value), unit)
        else:
            name, value = map(str.strip, line.split(":", 1))
            match = re.match(r"([0-9.]+)\s*(\w*)", value)
            if match:
                value_num, unit = match.groups()
                return Metric(name, float, float(value_num), unit)

        return None

    def _kernel_timing_df(self, lines: List[str]) -> pd.DataFrame:
        data = "\n".join(lines)
        return pd.read_csv(io.StringIO(data), sep=r"\s{2,}", header=0, engine="python")

    def get_logfile(self) -> Union[str, os.PathLike]:
        """
        Returns the logfile associated with the metrics.
        """
        return self.logfile

    def get_metrics(self) -> Optional[List[Metric]]:
        """
        Returns the list of metrics.
        """
        return self.metrics
