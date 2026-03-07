"""plot relative errors for cumulants
"""

from .checkpoint_plots import PlotUtils

if __name__=="__main__":

    plot_utils = PlotUtils()

    plot_utils.make_cumulant_error_plot_from_csv()
