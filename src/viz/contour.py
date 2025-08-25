import numpy as np 
import matplotlib.pyplot as plt 
import torch

import pdb
##creates the contour plot showing the full joint pdf

def generate_mesh(x_samples,qmax=6):
    """
    generates a mesh in two dimensional phase space to evaluate and plot the joint distribution 
    
    args:
        x_samples:
            number of samples in the space discretisation
    """

    #pmax,qmax = 4,4
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
def compute_joint(t,net,
                    testing_samples,
                    x_samples,
                    boundary_cond):
    """
    computes an estimate of the joint distribution based on the method of 2411.08518

    args:
        t:
            time at which to evaluate the distribution
        testing_samples:
            number of independent monte carlo trajectories at each point in the mesh
        x_samples: 
            number of points in the discretisation of one axis (ie for two dimensional phase space we get x_samples**2 points)
        boundary_cond:
            boundary conditions class specifying the problem
    """

    net.eval()
    
    t_vec = np.array(boundary_cond.get_reference_timegrid(t+.0001))
    
    #start the trajectories
    xinit = torch.tensor(generate_mesh_mc(x_samples,testing_samples),
                            requires_grad=False,
                            dtype=torch.float32)
    #reset girsanov
    girsanov = torch.zeros((testing_samples,x_samples*x_samples,1))

    for i in range(1,len(t_vec)):#range(0,len(t_vec)):

        dt = np.abs(t_vec[-i-1]-t_vec[-i])
        
        dw = np.sqrt(dt)*torch.randn((testing_samples,x_samples*x_samples,1))
        nn_input = torch.dstack((t_vec[-i]*torch.ones((testing_samples,x_samples**2,1),dtype=torch.float32),
                                    xinit[...,[0]]))
        du = torch.hstack([net(nn_input[:,i,:]) for i in range(x_samples**2)])[...,np.newaxis]

        #underdamped dynamics
        #backward (T-t0) evolution in t0
        xinit = boundary_cond.girsanov_scheme(xinit,du,-dt,-dw,t_vec[i])
        
        #remove escaped particles
        xinit[xinit>30] = torch.nan
        xinit[xinit<-30] = torch.nan

        #compute girsanov factor
        girsanov = girsanov + (np.sqrt(1/boundary_cond.sigma0)*(xinit[...,[1]]*dw) + (dt/(boundary_cond.sigma0**2))*torch.square(xinit[...,[1]]))

    return np.nanmean(np.multiply(boundary_cond.p_initial(xinit[...,[0]],xinit[...,[1]]),np.exp(-girsanov)),axis=0)


@torch.no_grad
def plot_contour(fig_joint_distributions_meshgrid,
                    gs_joint_distributions,
                    net,
                    testing_samples,
                    x_samples,
                    boundary_cond,
                    plot_times):

    #reset plot index
    plot_index = 0

    for t in plot_times:

        joint_out = compute_joint(t,net,
                    testing_samples,
                    x_samples,
                    boundary_cond)
        #np.nanmean(np.multiply(boundary_cond.p_initial(xinit[...,[0]],xinit[...,[1]]),np.exp(-girsanov)),axis=0)

        
        plotter.joint_distributions_scatter(fig_joint_distributions_meshgrid,
                                                gs_joint_distributions,
                                                plot_index,
                                                joint_out,
                                                x_samples,
                                                t,
                                                vmax=vmax)
        plot_index += 1
    ####



def format_plot(net,
                testing_samples,
                x_samples,
                boundary_cond,
                plot_times):

    # Plotting the distributions -initialise the figure object & creates the gridspec
    fig_joint_distributions_meshgrid = plt.figure(figsize=(15,10))
    gs_joint_distributions = fig_joint_distributions_meshgrid.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    #compute data and create subplots
    plot_contour(fig_joint_distributions_meshgrid,
                        gs_joint_distributions,
                        net,
                        testing_samples,
                        x_samples,
                        boundary_cond,
                        plot_times)

    #adjust spacing
    fig_joint_distributions_meshgrid.subplots_adjust(wspace=1.1,# The width of the padding between subplots, as a fraction of the average Axes width.
                                                    hspace=1.1# The height of the padding between subplots, as a fraction of the average Axes height.
                                                    )
    
    #add colourbar
    plotter.make_colourbar(fig_joint_distributions_meshgrid,vmax)
    plotter.make_legend(fig_joint_distributions_meshgrid)
    fig_joint_distributions_meshgrid.savefig(plotter.subdir+"/test_formatting.png",bbox_inches="tight")
    plt.close(fig_joint_distributions_meshgrid)


if __name__=="__main__":
    import os 

    from ..model.nn import NeuralNetwork
    from ..sde.boundary import UnderdampedNoMomentum
    from .plots import PlotFuns
    from ..utils.parser import girsanov_plot_params
    
    girs_params = girsanov_plot_params()
    testing_samples = girs_params.mcsamples
    x_samples = girs_params.xsamples
    vmax = 0.5

    new_folder = "tests/20250523-174017"

    boundary_cond = UnderdampedNoMomentum()
    net = NeuralNetwork(boundary_cond)
    net.load_checkpoint(new_folder+"/checkpoints/net_checkpoint_2000.pt")

    plotter = PlotFuns(boundary_cond)
    plotter.setresultsdir(new_folder)
    format_plot(net,
                    testing_samples,
                    x_samples,
                    boundary_cond,
                    [0,1,2,3,4,5])
