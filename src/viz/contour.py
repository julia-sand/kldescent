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
        
        du = np.zeros_like(xinit[...,[0]])
        
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




def joint_distributions_scatter_panel(self,
                                        ax,
                                        ax_pmarginal,
                                        ax_qmarginal,
                                        joint_out,
                                        xs,time,vmax=0.5):
    """
    Plots one panel of the joint distribution, where the 
    center shows the joint and marginals are estimated

    args:
      ax: artist
        axes where to plot. can be a gridspec subplot
      joint_out: arr
        joint distribution (n**2,1) as a numpy array
      xs: arr
        2d input data shaped (n**2,2) as numpy arr
      vmax: float
        maximum value for the scatter plots color scale
    """

    #pdb.set_trace()
    n = joint_out.shape[0]
    
    #pdb.set_trace()
    joint_out_temp = joint_out.reshape((int(np.sqrt(n)),int(np.sqrt(n))))
    Q = xs[...,[0]].reshape((int(np.sqrt(n)),int(np.sqrt(n))))
    p_init = np.sort(Q[[0],:].flatten()) #one axis 
    dx = np.abs(p_init[1]-p_init[0])

    #compute marginals
    qmarg = np.trapz(joint_out_temp,x = p_init,axis=0)#np.sum(joint_out_temp,axis=0) #np.trapz(joint_out_temp,x = p_init,axis=0)
    pmarg = np.trapz(joint_out_temp,x = p_init,axis=1)
    qnorm = self.get_norm(p_init, qmarg)
    pnorm = self.get_norm(p_init, pmarg)

    #plot and format marginals
    self.plot_marginal_joint(ax_qmarginal,p_init,qmarg/qnorm)
    self.plot_marginal_joint(ax_pmarginal,pmarg/pnorm,p_init)

    #plot marginals
    x,ptx = self.referencesol.get_quantity(np.round(time,6),quantity = "marginal",data="perturbative underdamped")
    ax_qmarginal.plot(ptx,x,color="gray",linestyle="dashed")

    #plot joint distribution
    ax.scatter(xs[...,0]/(qnorm*pnorm),xs[...,1]/(qnorm*pnorm),c=joint_out_temp.T.flatten()
              ,cmap="Blues",norm="log",vmin=0.001,vmax=vmax)
    


def scatter_panel_layout(self,ax,time=None):

    if time ==None:
        time = self.boundary.params["Tf"]

    plot_title_value = round(time/self.boundary.params["Tf"],2)
    ax.set_title(f"t={plot_title_value}Tf")

    #make inset axes to show marginal distributions
    ax_pmarginal = ax.inset_axes([0, 1.05, 1, 0.6])
    ax_qmarginal = ax.inset_axes([-0.65, 0, 0.6, 1])
    ax_qmarginal.invert_xaxis()
    ax_qmarginal.set_ylim((-self.qmax,self.qmax))

    ax_pmarginal.set_xlim((-self.qmax,self.qmax))

    self.add_boundary_conditions_scatter_panel(ax_qmarginal,ax_pmarginal,time)

    ###ticks formatting and labelling
    ax_pmarginal.yaxis.tick_right()
    ax.yaxis.tick_right()

    ax_pmarginal.tick_params(
        axis='y',
        which='both',
        length=5,
        rotation=-20,
        pad = -3)

    ax_qmarginal.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',
        rotation=70,
        length =5,
        pad = -3,
        labelleft=False)
    ax_qmarginal.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      #both major and minor ticks
        left=False,
        labelleft=False)

    #ax.set_xlim((-self.qmax,self.qmax))
    #ax.set_ylim((-self.qmax,self.qmax))

    ax_pmarginal.set_xticklabels([])

    ax.set_xlabel(r"$p$")
    ax_qmarginal.set_xlabel(r"$\rho_t(q)$")

    ax.set_ylabel(r"$q$")
    ax_pmarginal.set_ylabel(r"$\rho_t(p)$")
    ax.yaxis.set_label_position("right")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-2.5,0,2.5])

    return ax, ax_pmarginal,ax_qmarginal




def plot_joint_panel(self,ax,time,jointout,xout):
    ax, ax_pmarginal,ax_qmarginal = scatter_panel_layout(ax,time)

    joint_distributions_scatter_panel(ax,
                                        ax_pmarginal,
                                        ax_qmarginal,
                                        jointout,
                                        xout,time,vmax=0.5)

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
