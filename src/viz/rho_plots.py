from .checkpoint_plots import PlotUtils

if __name__=="__main__":

    plot_utils = PlotUtils()

    for plottime in [1,2,3,4,5]:
        #plot_utils.checkpoint_plots_rho()
        plot_utils.make_intermediate_histograms(plottime = plottime)