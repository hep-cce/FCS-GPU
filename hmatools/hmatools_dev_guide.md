# Developer Guide

This guide is intended to help collaborators and contributors. General usage information can be
obtained in `README.md`.

# HMATools

HMATools is a Python package to parse and plot logfiles produced by [FastCaloSim](https://github.com/hep-cce/FCS-GPU) and ultimately other HEPMiniApps (HMA).

## Repository Structure

- **LICENSE**: License file for the project.
- **README.md**: Project documentation.
- **hmatools.Dockerfile**: Dockerfile for building the project ina a containerized environment.
- **pyproject.toml**: Configuration file for building and packaging the Python project.
- **requirements.txt**: List of dependencies required by the project.

- **python/**: Main source directory for the HMATools package.
  - `__init__.py`: Marks the directory as a Python package.
  - `hma_metrics.py`: Module for handling metrics-related functionality.
  - `hma_plot.py`: Module for plotting results.
  - `hma_result.py`: Module for parsing and processing HMA results.
  - `hma_run.py`: Module for managing individual HMA runs.
  - `plotparser.py`: Parser module for plotting data.
  - `postprocparser.py`: Parser module for postprocessing data.
  - `run_info.py`: Module for handling run information like system, hardware, timestamp, etc.
  - `util.py`: Utility functions used across the package.

- **scripts/**: Shell scripts for various automation tasks.
  - `build_image.sh`: Script to build the hmatools Docker image.
  - `plot.sh`: Generic script to run plotting tasks inside containers.
  - `postprocess.sh`: Generic script to run postprocessing tasks inside containers.
  - `run_image.sh`: Script to run the hmatools Docker image, mount input/output and run scripts.

- **setup.py**: Setup script for installing the package.

- **tests/**: Test suite for the project.
  - `__init__.py`: Marks the directory as a Python package.
  - **test_data/**: Directory containing test data.
    - **Log Files - Run FastCaloSim/**: Example log files for testing.
    - `fcs_results.json`: JSON file with test results.
    - `fcs_results.png`: Plot image generated from test results.
  - `test_fcs_result.py`: Test script for validating `hma_result.py` and `hma_plot.py`.

## Main modules

The core functionalities of postprocessing log files and plotting results are handled by `hma_result.py` and `hma_plot.py`, respectively. Hence, the following contains a detailed break down of these two modules.

### hma_result.py

The `hma_result.py` module provides functionality to handle HMA results, primarily through the `HmaResult` class. Below is an overview of the key components of this module:

#### Classes and Functions

##### `HmaResult`
The `HmaResult` class represents the results of one or more HMA runs, consisting of multiple `HmaRun` instances.

- **`__init__(input_path: Union[str, os.PathLike])`**:
  Initializes the `HmaResult` object with the path to an input directory or file. Depending on whether the input is a directory or a single file, it calls the appropriate private method to retrieve the results.

- **`_get_results_from_directory() -> List[HmaRun]`**:
  A private method that reads log files from the specified input directory, initializes `HmaRun` objects for each log file, and stores them in a list. It raises a `ValueError` if no valid results are found.

- **`_get_results_from_file(file_path: Union[str, os.PathLike]) -> List[HmaRun]`**:
  A private method that reads a single log file, initializes a `HmaRun` object, and stores it in a list.

- **`dump_to_json(filename: Optional[str] = None) -> None`**:
  Dumps the results to a JSON file. If no filename is specified, it saves the file in the current working directory with a timestamp-based name. The function raises a `ValueError` if the output path is invalid.

- **`get_results() -> List[HmaRun]`**:
  Returns the list of `HmaRun` objects representing the results.

- **`get_input_dir() -> Union[str, os.PathLike]`**:
  Returns the path to the input directory.

##### `json_serial`
A helper function for JSON serialization of various object types that are not serializable by default (e.g., `datetime`, `pd.DataFrame`, `HmaRun`).

##### `main(input_path: str, output: str) -> None`
A standalone function that initializes a `HmaResult` object and triggers the postprocessing by calling `dump_to_json()`.

### Example Usage in Code

If you are integrating `hma_result.py` into a larger workflow, you might use it as follows:

```python
from hmatools.hma_result import HmaResult

# Initialize the HmaResult with a directory or a single log file
result = HmaResult("./logs")

# Retrieve the results
runs = result.get_results()

# Optionally, dump the results to a JSON file
result.dump_to_json("./output/results.json")
```

## hma_plot.py

The `hma_plot.py` module is part of the HMATools package and is designed for plotting results parsed from logfiles produced by FastCaloSim and other HEPMiniApps (HMA). Below is an overview of its key components:


#### Classes and Functions

##### `HmaPlot`
The `HmaPlot` class provides functionality to load and plot HMA results. It includes methods for reading input data, preparing the data for plotting, and generating comparative plots of kernel run times.

- **`__init__(input_path: str) -> None`**:
  Initializes the `HmaPlot` object with a specified input path, which can be a file or directory. Reads and deserializes the HMA data, extracts kernel information, and prepares a DataFrame for plotting.

- **`read_hma_input() -> List[Dict[str, Any]]`**:
  Reads and deserializes HMA result data from a given file or directory.

- **`get_kernel_data() -> Tuple[List[pd.DataFrame], List[Tuple[str, int]]]`**:
  Extracts kernel timing data and launch counts from the HMA results, returning a list of DataFrames and a list of launch counts.

- **`prepare_dataframe() -> pd.DataFrame`**:
  Combines multiple kernel result DataFrames into one, standardizing kernel names and preparing the data for plotting.

- **`plot(save_plot: bool = True, filename: Optional[str] = None) -> None`**:
  Generates bar plots comparing the total elapsed run time and average kernel run time for different technologies and kernels. It includes error bars for the average run time and can either save the plot to a file or display it.

- **`save_or_show_plot(save_plot: bool = True, filename: Optional[str] = None) -> None`**:
  Saves the generated plot to a file if `save_plot` is `True`, otherwise displays the plot.

##### `json_deserial(dct: Dict[str, Any]) -> Union[Dict[str, Any], Metric]`
A helper function that deserializes JSON objects into Python objects, including custom `Metric` objects used in HMA results.

##### `main(input_path: str, output: str) -> None`
A standalone function that initializes an `HmaPlot` object and triggers the plotting process, saving the output to the specified file.

### Example Usage in Code

If you are integrating `hma_plot.py` into a larger workflow, you might use it as follows:

```python
from hmatools.hma_plot import HmaPlot

# Initialize the HmaPlot with a directory or a single result file
plot = HmaPlot("./results")

# Generate and save the plot
plot.plot(save_plot=True, filename="./output/plot.png")
```
