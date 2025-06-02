import numpy as np 
import matplotlib.pyplot as plt 
import torch

##creates the contour plot showing the full joint pdf

def generate_mesh(x_samples,testing_samples):
    #set up the time grid things
    P,Q = np.meshgrid(np.linspace(-4,4,x_samples),
                        np.linspace(-4,4,x_samples))

    return np.repeat(np.column_stack((P.reshape(x_samples*x_samples),Q.reshape(x_samples*x_samples)))[np.newaxis, ...], testing_samples, axis=0)
    #np.dstack([np.column_stack((P.reshape(x_samples*x_samples),Q.reshape(x_samples*x_samples)))]*testing_samples)

@torch.no_grad
def plot_contour(fig_joint_distributions_meshgrid,
                    gs_joint_distributions,
                    net,
                    testing_samples,
                    x_samples,
                    boundary_cond,
                    plot_times):

    net.eval()
    #pdb.set_trace()
    t_vec = np.array(boundary_cond.get_reference_timegrid(boundary_cond.Tf+.0001))
    
    #reset plot index
    plot_index = 0

    for t in plot_times:

        start_index = np.where(t_vec==t)[0][0]

        #start the trajectories
        xinit = torch.tensor(generate_mesh(x_samples,testing_samples),
                                requires_grad=False,
                                dtype=torch.float32)
        #reset girsanov
        girsanov = torch.zeros((testing_samples,x_samples*x_samples,1))

        for i in range(0,start_index):

            dt = np.abs(t_vec[start_index-i]-t_vec[start_index-i-1])
            
            dw = np.sqrt(dt)*torch.randn((testing_samples,x_samples*x_samples,1))
            nn_input = torch.dstack((t_vec[start_index-i]*torch.ones((testing_samples,x_samples**2,1),dtype=torch.float32),
                                        xinit[...,[0]]))
            du = torch.hstack([net.forward(nn_input[:,i,:]) for i in range(x_samples**2)])[...,np.newaxis]

            #underdamped dynamics
            #backward (T-t0) evolution in t0
            xinit = boundary_cond.girsanov_scheme(xinit,du,-dt,-dw,t_vec[i])
            
            #remove escaped particles
            xinit[xinit>30] = torch.nan
            xinit[xinit<-30] = torch.nan

            #compute girsanov factor
            girsanov = girsanov + (np.sqrt(1/boundary_cond.sigma0)*(xinit[...,[1]]*dw) + (dt/(boundary_cond.sigma0**2))*torch.square(xinit[...,[1]]))

        joint_out = np.nanmean(np.multiply(boundary_cond.p_initial(xinit[...,[0]],xinit[...,[1]]),np.exp(-girsanov)),axis=0)

        
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

    testing_samples = 10
    x_samples = 8
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
