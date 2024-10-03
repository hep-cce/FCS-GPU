"""
Module for parsing command line arguments and invoking the postprocessing main function.

This module provides a command line interface for postprocessing HMA results.
It parses the input and output file paths from the command line arguments and
calls the main postprocessing function from the `hma_result` module.
"""

import argparse

from hma_result import main as postprocess_main


def parse_args():
    """Parse command line arguments for the postprocessing script."""
    plot_parser = argparse.ArgumentParser(
        description="Postprocess command line arguments."
    )
    plot_parser.add_argument("-i", "--input", required=True, help="Input dir/file")
    plot_parser.add_argument("-o", "--output", required=True, help="Output dir/file")

    args = plot_parser.parse_args()
    return args


def main():
    """Main function for parsing command line arguments and invoking postprocessing."""
    args = parse_args()
    postprocess_main(args.input, args.output)
