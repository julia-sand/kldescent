"""plot relative errors for cumulants
"""
from .checkpoint_plots import PlotUtils
from ..utils.parser import fetch_results_folder_from_cmd

def plot_errs():
    folder = fetch_results_folder_from_cmd()

    