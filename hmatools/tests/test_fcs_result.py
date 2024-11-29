import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from hma_result import HmaResult
from hma_plot import HmaPlot

def test_get_results():
    input_dir = os.path.join(os.path.dirname(__file__), "test_data/Log Files - Run FastCaloSim")
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist"
    num_files = len([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

    result = HmaResult(input_dir)
    results = result.get_results()
    assert len(results) == num_files

def test_dump_to_json(tmp_path):
    input_dir = os.path.join(os.path.dirname(__file__), "test_data/Log Files - Run FastCaloSim")
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist"
    result = HmaResult(input_dir)

    # Dump results to JSON
    output_file = tmp_path / 'fcs_results.json'
    result.dump_to_json(output_file)

    # Verify the JSON file
    assert output_file.exists()
    with open(output_file, 'r') as f:
        data = json.load(f)

    assert len(data) == 6    # Create a temporary directory for output

def test_plot(tmp_path):
    # Create the JSON file in the temporary directory
    input_dir = os.path.join(os.path.dirname(__file__), "test_data/Log Files - Run FastCaloSim")
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist"
    result = HmaResult(input_dir)
    json_file = tmp_path / 'fcs_results.json'
    result.dump_to_json(json_file)

    # Now create the plot using the JSON file
    plot_file = tmp_path / "fcs_results.png"
    plot = HmaPlot(tmp_path)
    plot.plot(save_plot=True, filename=plot_file)

    # Verify the plot
    assert plot_file.exists()