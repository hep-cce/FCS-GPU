# GitHub Actions Workflows Overview

## 1. Run FastCaloSim Benchmarking (`benchmarking.yaml`)

The `benchmarking.yaml` workflow is designed to automate the benchmarking of FastCaloSim on different environments, including Perlmutter and Exalearn. Steps:

- **Build and Push Images**:
  - This step is optional and controlled via workflow dispatch (`run_build` input).
  - If triggered, the job builds and pushes Docker images using a self-hosted runner, and logs are uploaded for review.

- **Run FastCaloSim**:
  - This job runs FastCaloSim on different platforms using a matrix strategy to handle multiple configurations (Perlmutter and Exalearn).
  - It sets up environment variables specific to each runner and executes the simulation scripts.
  - Log files from each run are compressed into tarballs and uploaded for later use.

- **Postprocess Log Files**:
  - This job depends on the completion of the `run` job.
  - It downloads log files from the previous runs, decompresses them, and runs a postprocessing script using HMATools.
  - The processed logs and resulting JSON files are uploaded as artifacts.

- **Plot Results**:
  - This job depends on the completion of the `postprocess` job.
  - It downloads the processed JSON files and runs a plotting script using HMATools.
  - The generated plots and associated logs are uploaded for analysis.


## 2. Python Code Checks (`Python_CI.yaml`)

The `Python_CI.yaml` workflow is a Continuous Integration (CI) pipeline focused on ensuring code quality and correctness in the HMATools project. It includes the following steps:

- **Checkout Code**:
  - The repository is checked out to the runner for subsequent operations.

- **Set Up Python**:
  - Python 3.11 is installed and configured for the environment.

- **Install Dependencies**:
  - All necessary Python packages, including project dependencies and development tools like `black`, `flake8`, `pylint`, `mypy`, and `pytest`, are installed.

- **Code Formatting with Black**:
  - The codebase is checked for compliance with `black` formatting standards.

- **Linting with Flake8 and Pylint**:
  - Code is linted to catch syntax errors, enforce coding standards, and assess code complexity.

- **Type Checking with Mypy**:
  - Static type checking is performed to catch type-related errors before runtime.

- **Run Tests**:
  - Unit tests are executed using `pytest` to ensure the functionality is working as expected.
