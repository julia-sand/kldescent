import numpy as np 
import matplotlib.pyplot as plt 
import torch

from . import plots

##creates the contour plot showing the full joint pdf

def generate_mesh(x_samples,qmax=6):
    """
    generates a mesh in two dimensional phase space to evaluate and plot the joint distribution 
    
    args:
        x_samples:
            number of samples in the space discretisation
    """

    p_init = np.linspace(-qmax,qmax,x_samples)
    q_init = np.linspace(-qmax,qmax,x_samples)
    
    Q,P = np.meshgrid(q_init,p_init)
    return Q,P

def generate_mesh_mc(x_samples,testing_samples):
    """
    generates a mesh in two dimensional phase space and copies it to allow computing parallel montecarlo 
    trajectories from the same starting point

    args:
        x_samples:
            number of samples in the space discretisation
        testing_samples:
            number of copies of the meshgrid 
    """

    Q,P = generate_mesh(x_samples)

    return np.repeat(np.column_stack((Q.reshape(x_samples*x_samples),P.reshape(x_samples*x_samples)))[np.newaxis, ...], testing_samples, axis=0)

@torch.no_grad
def plot_contour(net,
                    testing_samples,
                    x_samples,
                    plotter,
                    plot_times):

    nplots = len(plot_times)
    ncols = int(np.ceil(nplots/2))

    #initialise the figure object & creates the gridspec
    fig_joint_distributions_meshgrid = plt.figure(figsize=(15,10))
    gs_joint_distributions = fig_joint_distributions_meshgrid.add_gridspec(2, ncols)

    for t in enumerate(plot_times):

        #define the axis 
        ax = plt.subplot(gs_joint_distributions[xind,yind])

        joint_out = compute_joint(t,net,
                                    testing_samples,
                                    x_samples,
                                    boundary_cond)
        
        plotter.plot_joint_panel(ax,t,jointout,xout)       

        plot_index += 1


def format_and_save_plot(fig,plotter):
    """Adds legend, colorbar and adjusts spacing for the figure
    saves the plot into plotter location

    args: 
        fig : figure
            generated scatter plot 
        plotter : class
            general plotting parameters
    """

    #add legend and colourbar
    plotter.make_colourbar(fig,vmax)
    plotter.make_legend(fig)
    
    fig.savefig(plotter.subdir+"/test_formatting.png",bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    import os 

    from ..model.nn import NeuralNetwork
    from ..sde.boundary import UnderdampedNoMomentum
    from .plots import PlotFuns
    from ..utils.parser import girsanov_plot_params
    
    girs_params = girsanov_plot_params()
    testing_samples = girs_params["mcsamples"]
    x_samples = girs_params["xsamples"]
    vmax = 0.5

    boundary_cond = UnderdampedNoMomentum()
    #net = NeuralNetwork(boundary_cond)
    
    plotter = PlotFuns(boundary_cond)

    format_plot(None,
                    testing_samples,
                    x_samples,
                    boundary_cond,
                    [0,1,2,3,4,5])
