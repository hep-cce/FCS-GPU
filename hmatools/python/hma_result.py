"""
Functionality to handle HMA results:

- HmaResult class which is responsible for initializing results from a given input directory,
retrieving these results, and dumping them to a JSON file
- Helper function for JSON serialization of various object types
"""

import os
import json

from datetime import datetime
from typing import Union, Optional, List

import pandas as pd

from hma_run import HmaRun
from util import get_current_timestamp


def json_serial(obj: object) -> Union[str, dict]:
    """JSON serializer for objects not serializable by default"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_json()
    if isinstance(obj, type):
        return str(obj)
    if isinstance(obj, HmaRun):
        return obj.to_dict()
    raise TypeError(f"Type {type(obj)} not serializable")


class HmaResult:
    """Representation of one HmaResult, consisting of multiple HmaRuns"""

    def __init__(self, input_path: Union[str, os.PathLike]) -> None:
        """Initialize hma_results with the path to the input directory or file."""
        if os.path.isdir(input_path):
            self.input_dir = input_path
            self.results = self._get_results_from_directory()
        elif os.path.isfile(input_path):
            self.input_dir = os.path.dirname(input_path)
            self.results = self._get_results_from_file(input_path)
        else:
            raise ValueError(
                f"The provided input path '{input_path}' is neither a file nor a directory."
            )

    def _get_results_from_directory(self) -> List[HmaRun]:
        """Get the individual HmaRun objects from the input directory"""
        results = []
        input_files = [
            x
            for x in os.listdir(self.input_dir)
            if x.endswith(".txt") and x.startswith("run_log_")
        ]
        input_files.sort()
        input_files = [os.path.join(self.input_dir, file) for file in input_files]
        for logfile in input_files:
            print(f"Reading {logfile}")
            run = HmaRun(logfile)
            results.append(run)
        if not results:
            raise ValueError(f"No HMA results found in directory {self.input_dir}")
        return results

    def _get_results_from_file(
        self, file_path: Union[str, os.PathLike]
    ) -> List[HmaRun]:
        """Get the HmaRun object from a single file"""
        print(f"Reading {file_path}")
        run = HmaRun(file_path)
        return [run]

    def dump_to_json(self, filename: Optional[str] = None) -> None:
        """Dump the results to a JSON file in the specified file or a default directory"""

        if filename is None:
            current_dir = os.getcwd()
            json_filename = os.path.join(
                current_dir, f"results_{get_current_timestamp()}.json"
            )
        elif os.path.isdir(filename):
            json_filename = os.path.join(
                filename, f"results_{get_current_timestamp()}.json"
            )
        elif os.path.isdir(os.path.dirname(filename)):
            json_filename = filename
        else:
            raise ValueError(f"Invalid output path: {filename}")

        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, sort_keys=True, default=json_serial)

        print(f"Results successfully saved to {json_filename}")

    def get_results(self) -> List[HmaRun]:
        """Public method to get the results"""
        return self.results

    def get_input_dir(self) -> Union[str, os.PathLike]:
        """Public method to get the input directory path"""
        return self.input_dir


def main(input_path, output):
    """Main function for postprocessing HMA results."""
    print(f"Postprocessing: input_path={input_path}, output={output}")
    result = HmaResult(input_path)
    result.dump_to_json(output)
