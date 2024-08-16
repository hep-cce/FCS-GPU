"""
Module for parsing command line arguments and invoking the plotting main function.

This module provides a command line interface for plotting HMA results.
It parses the input and output file paths from the command line arguments and
calls the main plotting function from the `hma_plot` module.
"""

import argparse
from hma_plot import main as plot_main


def parse_args():
    """Parse command line arguments for the plotting script."""
    plot_parser = argparse.ArgumentParser(description="Plot command line arguments.")
    plot_parser.add_argument("-i", "--input", required=True, help="Input file")
    plot_parser.add_argument("-o", "--output", required=True, help="Output file")

    args = plot_parser.parse_args()
    return args


def main():
    """Main function for parsing command line arguments and invoking plotting."""
    args = parse_args()
    plot_main(args.input, args.output)
