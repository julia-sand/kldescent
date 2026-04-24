from kldescent.viz.checkpoint_plots import PlotUtils
from kldescent.utils.parser import fetch_results_folder_from_cmd

if __name__=="__main__":

    folder = fetch_results_folder_from_cmd()
    plot_utils = PlotUtils(folder=folder)

    #plot_utils.perturbative_plots_du(plottimes =[0,1,1.2,2,2.4,3,3.8,4,4.7,5])
    plot_utils.make_contour_plot2(plottimes=[0,1,2,3,4,5])
    #plot_utils.make_sde_plots(2048,usebenchmark=False)