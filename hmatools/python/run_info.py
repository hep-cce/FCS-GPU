"""
Module for parsing and extracting run information from log files.
"""

import os
import re
import json

from typing import Union, Optional, Tuple, Dict
from datetime import datetime
from util import remove_trailing_commas, parse_csv_data


class RunInfo:
    """
    Class for parsing and storing run information from a log file.
    """

    def __init__(self, logfile: Union[str, os.PathLike], info_data: str) -> None:
        runner_label, description, timestamp = self._split_log_name(logfile)
        self.runner_label = runner_label
        self.image_info = self._parse_description(description)
        self.timestamp = self._parse_timestamp(timestamp)
        self.run_cmd = self._parse_run_cmd(info_data)
        self.system_info = self._parse_system_info(info_data)
        self.cpu_info = self._parse_cpu_info(info_data)
        self.nvidia_info = self._parse_nvidia_info(info_data)

    def _split_log_name(self, logfile: Union[str, os.PathLike]) -> Tuple[str, str, str]:
        tag, _ = os.path.splitext(os.path.basename(logfile))
        if tag.startswith("run_log_"):
            tag = tag[9:]
        tags = tag.split("_")
        if len(tags) == 2:
            description, timestamp = tags
            runner_label = ""
        elif len(tags) == 3:
            runner_label, description, timestamp = tag.split("_")
        else:
            print(f"Could not parse log name {tag}")
            runner_label, description, timestamp = "", "", ""
        return runner_label, description, timestamp

    def _parse_description(self, description: str) -> Dict[str, str]:
        # Assuming the description is in the format "image_type-root_version-image_tag"
        description_parts = description.split("-")
        for i, part in enumerate(description_parts):
            if part[0].isdigit():
                image_info = {
                    "image_type": "-".join(description_parts[:i]),
                    "root_version": part,
                    "image_tag": "-".join(description_parts[i + 1 :]),
                }
                return image_info
        print(f"Could not parse description {description}")
        return {"image_type": "", "root_version": "", "image_tag": ""}

    def _parse_timestamp(self, timestamp: str) -> Optional[datetime]:
        timestamp_match = re.search(r"\d{14}", timestamp)
        if timestamp_match:
            timestamp_str = timestamp_match.group()
            parsed_timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
            return parsed_timestamp
        return None

    def _parse_run_cmd(self, data: str) -> Optional[str]:
        command_match = re.search(r"runTFCSSimulation\s.*", data)
        if command_match:
            return command_match.group(0)
        return None

    def _parse_system_info(
        self, file: str
    ) -> Optional[Dict[str, Union[str, int, float]]]:
        json_string = str()
        hits = 0
        for line in file.splitlines():
            if "{" in line:
                hits += 1

            if hits >= 1:
                json_string += line

            if "}" in line:
                hits -= 1

        if json_string:
            json_string = remove_trailing_commas(json_string)
            return json.loads(json_string)
        print(
            f"No system information found for {self.image_info['image_type']} "
            f"with tag {self.image_info['image_tag']}"
        )
        return None

    def _parse_cpu_info(self, file: str) -> Dict[str, Union[str, int, float]]:
        cpu_info = {}
        patterns = {
            "Architecture": r"Architecture:\s+(.+)",
            "CPU op-mode(s)": r"CPU op-mode\(s\):\s+(.+)",
            "Address sizes": r"Address sizes:\s+(.+)",
            "Byte Order": r"Byte Order:\s+(.+)",
            "CPU(s)": r"CPU\(s\):\s+(\d+)",
            "On-line CPU(s) list": r"On-line CPU\(s\) list:\s+(.+)",
            "Vendor ID": r"Vendor ID:\s+(.+)",
            "Model name": r"Model name:\s+(.+)",
            "CPU family": r"CPU family:\s+(\d+)",
            "Model": r"Model:\s+(\d+)",
            "Thread(s) per core": r"Thread\(s\) per core:\s+(\d+)",
            "Core(s) per socket": r"Core\(s\) per socket:\s+(\d+)",
            "Socket(s)": r"Socket\(s\):\s+(\d+)",
            "Stepping": r"Stepping:\s+(\d+)",
            "Frequency boost": r"Frequency boost:\s+(.+)",
            "CPU max MHz": r"CPU max MHz:\s+([\d.]+)",
            "CPU min MHz": r"CPU min MHz:\s+([\d.]+)",
            "BogoMIPS": r"BogoMIPS:\s+([\d.]+)",
            "Flags": r"Flags:\s+(.+)",
            "Virtualization": r"Virtualization:\s+(.+)",
            "L1d cache": r"L1d cache:\s+(.+)",
            "L1i cache": r"L1i cache:\s+(.+)",
            "L2 cache": r"L2 cache:\s+(.+)",
            "L3 cache": r"L3 cache:\s+(.+)",
            "NUMA node(s)": r"NUMA node\(s\):\s+(\d+)",
            "NUMA node0 CPU(s)": r"NUMA node0 CPU\(s\):\s+(.+)",
            "NUMA node1 CPU(s)": r"NUMA node1 CPU\(s\):\s+(.+)",
            "Vulnerability Gather data sampling": r"Vulnerability Gather data sampling:\s+(.+)",
            "Vulnerability Itlb multihit": r"Vulnerability Itlb multihit:\s+(.+)",
            "Vulnerability L1tf": r"Vulnerability L1tf:\s+(.+)",
            "Vulnerability Mds": r"Vulnerability Mds:\s+(.+)",
            "Vulnerability Meltdown": r"Vulnerability Meltdown:\s+(.+)",
            "Vulnerability Mmio stale data": r"Vulnerability Mmio stale data:\s+(.+)",
            "Vulnerability Retbleed": r"Vulnerability Retbleed:\s+(.+)",
            "Vulnerability Spec rstack overflow": r"Vulnerability Spec rstack overflow:\s+(.+)",
            "Vulnerability Spec store bypass": r"Vulnerability Spec store bypass:\s+(.+)",
            "Vulnerability Spectre v1": r"Vulnerability Spectre v1:\s+(.+)",
            "Vulnerability Spectre v2": r"Vulnerability Spectre v2:\s+(.+)",
            "Vulnerability Srbds": r"Vulnerability Srbds:\s+(.+)",
            "Vulnerability Tsx async abort": r"Vulnerability Tsx async abort:\s+(.+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, file)
            if match:
                cpu_info[key] = match.group(1)

        return cpu_info

    def _parse_nvidia_info(self, file: str) -> Optional[Dict[str, str]]:
        pattern = re.compile(
            r"Executing: nvidia-smi --query-gpu=.*? --format=csv\n(.*?)(?=\nINFO -|\Z)",
            re.DOTALL,
        )
        match = pattern.search(file)

        if not match:
            print("No nvidia-smi output found in the logfile.")
            return None

        csv_data_block = match.group(1).strip()

        lines = csv_data_block.split("\n")
        if len(lines) < 2:
            print("Incomplete GPU information.")
            return None

        csv_data = "\n".join(lines)
        gpu_info = parse_csv_data(csv_data)
        return gpu_info

    def get_image_info(self) -> Dict[str, str]:
        """
        Get the image information.
        """
        return self.image_info

    def get_timestamp(self) -> Optional[datetime]:
        """
        Get the timestamp of the run.
        """
        return self.timestamp
