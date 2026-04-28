"""Create the contour plots showing the full joint pdf, where the pdf is computed 
either using a neural network interpolation, or using a Monte Carlo integration"""


import numpy as np 
import matplotlib.pyplot as plt 
import torch

from kldescent.sde.girsanovpdf import compute_pdf
from kldescent.viz.plots import *
 
@torch.no_grad
def generate_mesh(x_samples,qmax=6):
    """
    generate a mesh in two dimensional phase space to evaluate and plot the joint distribution 
    
    args:
        xsamples : int
            number of samples in the space discretisation
        qmax : float
            maximum value of the xaxis, symmetric in positive and negative
    """

    p_init = torch.linspace(-qmax,qmax,x_samples, requires_grad=False,dtype=torch.float32)
    #q_init = torch.linspace(-qmax,qmax,x_samples, requires_grad=False,dtype=torch.float32)
        
    return torch.meshgrid(p_init,p_init,indexing='ij')


@torch.no_grad
def compute_rho_girsanov(x_samples=40,boundary=None,net=None,mc_samples=64,Tf=None):
    """
    Computes an estimate of the final joint distribution using girsanov theorem
    under the input drift 
    
    args:
        dunet : class 
            neural network approximating the drift
        x_samples : int 
            number of space discretisation steps
        mc_samples : int
            number of MC samples. 
        Tf : float
            time for the plot
    returns:
        xout: 
            2d mesh of points of the statespace
        jointout:
            estimated joint distribution using girsanov theorem
    """
    
    #generate xmesh
    Q,P = generate_mesh(x_samples,qmax=4)
    xinit = torch.stack((P,Q), dim=-1).reshape(-1, 2)
    #torch.tensor(np.column_stack((Q.flatten(),P.flatten())),dtype=torch.float32,requires_grad=False)
    
    with torch.no_grad():
        self.net.eval()

        #intiialise pdf 
        pdf_out = torch.zeros_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32)

        pdf_out = compute_pdf(xinit.clone(),net,
                                    self.boundary.sigma0,
                                    self.boundary.params["epsilon"],
                                    self.boundary.p_initial,
                                    Tf).cpu().numpy()
    return xinit.cpu().numpy(),pdf_out.cpu().numpy()

@torch.no_grad
def distribution_plot(plot_loc,model,Tf,epoch=1,qmax=5):

    #compute marginals 
    xx, yy = torch.meshgrid(torch.linspace(-qmax,qmax, 100), torch.linspace(-qmax,qmax, 100), indexing="xy")
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    
    #find probabilities 
    pdf_temp = torch.exp(model.log_prob(zz))
    pdf_temp[torch.isnan(pdf_temp)] = 0

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(25)

    xout,jointout = zz,pdf_temp.detach().clone().numpy().flatten()

    plot_joint_panel(plt.gca(),5,jointout,xout)

    fig.tight_layout()
    fig.savefig(plot_loc+f"/contour_distribution_{epoch}.png",bbox_inches="tight")        

    plt.close()

    fig1 = plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, pdf_temp.view(*xx.shape).detach().clone().numpy(), cmap='coolwarm')
    fig1.savefig(plot_loc+"/pdf_distribution.png",bbox_inches="tight") 
    plt.close()

def get_norm(x,y):
    idx = np.argsort(x)
    return np.trapezoid(y[idx],x[idx])

def add_boundary_conditions_scatter_panel(ax_pmarginal,ax_qmarginal,time,boundary):

    if time==0:
        ax_qmarginal.fill_between(
                boundary.q_marginal_initial_np(np.linspace(-boundary.qmax,boundary.qmax,)),np.linspace(-boundary.qmax,boundary.qmax,),color="orange",label="Assigned Initial Distribution")
        
        ax_pmarginal.fill_between(np.linspace(-boundary.qmax,boundary.qmax,),
                normal_distribution(np.linspace(-boundary.qmax,boundary.qmax,),0,1),color="orange",label="Assigned Initial Distribution")

    elif time==boundary.params["Tf"]:
        ax_qmarginal.fill_between(
                boundary.q_marginal_final_np(np.linspace(-boundary.qmax,boundary.qmax,)),np.linspace(-boundary.qmax,boundary.qmax,),color="orange",label="Target Final Distribution")
        
        ax_pmarginal.fill_between(np.linspace(-boundary.qmax,boundary.qmax,),
                normal_distribution(np.linspace(-boundary.qmax,boundary.qmax,),0,1),color="darkorange",label="Target Final Distribution")

    else: 
        pass

def plot_marginal_joint(ax,x0_marginal,x0_init):

    ax.plot(x0_init,x0_marginal
                        ,label="Girsanov")


def scatter_panel_layout(ax,time,boundary=None):
    """format axes for the contour plot and add assigned boundary conditions"""

    #make inset axes to show marginal distributions
    ax_pmarginal = ax.inset_axes([0, 1.05, 1, 0.6])
    ax_qmarginal = ax.inset_axes([-0.65, 0, 0.6, 1])
    ax_qmarginal.invert_xaxis()

    ###ticks formatting and labelling
    ax_pmarginal.yaxis.tick_right()
    ax.yaxis.tick_right()

    if boundary is not None:  
        plot_title_value = round(time/boundary.params["Tf"],2)
        ax.set_title(f"t={plot_title_value}Tf")
        
        #set x axis limits
        ax.set_ylim((-boundary.qmax,boundary.qmax))
        ax.set_xlim((-boundary.qmax,boundary.qmax))
        ax_qmarginal.set_ylim((-boundary.qmax,boundary.qmax))
        ax_pmarginal.set_xlim((-boundary.qmax,boundary.qmax))

    else: 
        ax.set_title(f"t={time}")


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

    ax_pmarginal.set_xticklabels([])

    ax.set_xlabel(r"$p$")
    ax_qmarginal.set_xlabel(r"$\rho_t(q)$")
    ax_qmarginal.set_xlim(right=-0.01)#,left=0.6)

    ax.set_ylabel(r"$q$")
    ax_pmarginal.set_ylabel(r"$\rho_t(p)$")
    ax.yaxis.set_label_position("right")
    ax_pmarginal.set_ylim(bottom=-0.01,top=0.6)

    ax.set_aspect('equal', adjustable='box')
    #ax.set_xticks([-2.5,0,2.5])

    if boundary is not None:
        add_boundary_conditions_scatter_panel(ax_pmarginal,ax_qmarginal,time,boundary)

    return ax, ax_pmarginal,ax_qmarginal

def reshape_joint(joint_out):
    """reshape joint distribution for plotting"""

    n = joint_out.shape[0]

    return np.nan_to_num(joint_out.reshape((int(np.sqrt(n)),int(np.sqrt(n)))),nan=0,posinf=0,neginf=0)


def compute_marginals_from_joint(xs,joint_out):
    r"""Compute normalized position (q) and momentum (p) marginals from a joint distribution.

    Args:
    xs (ndarray): 
        Grid or coordinate values corresponding to the joint distribution.
    joint_out (ndarray): 
        Joint distribution data 

    Returns:
    pinit (ndarray): 
        Initial momentum distribution.
    q_coords (ndarray): 
        Coordinate values for the position (q) marginal.
    qmargnorm (ndarray): 
        Normalized position marginal distribution.
    pmargnorm (ndarray): 
        Normalized momentum marginal distribution.
    """

    joint_out_temp = reshape_joint(joint_out)

    p_init = np.unique(xs) 

    #compute marginals
    qmarg = np.trapezoid(joint_out_temp,x = p_init,axis=1)#np.sum(joint_out_temp,axis=0) #np.trapezoid(joint_out_temp,x = p_init,axis=0)
    pmarg = np.trapezoid(joint_out_temp,x = p_init,axis=0)

    #compute normalisation
    qnorm = get_norm(p_init, qmarg)
    pnorm = get_norm(p_init, pmarg)

    return p_init,qmarg/qnorm, pmarg/pnorm,qnorm*pnorm

def joint_distributions_scatter_panel(
                                    ax,
                                    ax_pmarginal,
                                    ax_qmarginal,
                                    joint_out,
                                    xs,time,vmax=0.5,
                                    benchmark=None):
    """
    Plot a single panel of a joint distribution.

    The central plot visualizes the joint distribution, while the marginal
    distributions are estimated from the same data.

    Args:
    ax (matplotlib.axes.Axes):
        The axes on which to draw the plot. Can be a standard axes
        or a subplot created from a GridSpec.

    ax_pmarginal (matplotlib.axes.Axes):
        The axes for marginal distribution of p

    ax_qmarginal (matplotlib.axes.Axes):
        The axes for marginal distribution of q

    joint_out (np.ndarray):
        Flattened joint distribution of shape (n**2,).
        Represents the values corresponding to each (q, p) pair.

    xs (np.ndarray):
        Input coordinates of shape (n**2, 2), where each row is a (q, p) pair.
        Assumes the points are generated via `np.meshgrid`, so that q and p
        values form a regular grid.

    time (float):
        Time of the evolution

    vmax (float):
        Upper limit for the color scale used in the joint distribution plot.Default 0.5
    
    benchmark (object):
        Stores the data for the benchmark solution. Default None, in which case benchmark values are not plotted.
    """

    p_init_out,qmarg_norm,pmarg_norm,normall = compute_marginals_from_joint(xs,joint_out)

    #plot and format marginals
    plot_marginal_joint(ax_qmarginal,p_init_out,qmarg_norm)
    plot_marginal_joint(ax_pmarginal,pmarg_norm,p_init_out)

    #if benchmark is not None:
    #plot marginals
    x,ptx = benchmark.get_benchmark_quantity(np.round(time,6),quantity = "underdamped marginal",data="perturbative underdamped",time_type="t0")
    ax_qmarginal.plot(ptx,x,color="gray",linestyle="dashed",label="Benchmark Predicted")

    #plot joint distribution
    ax.scatter(xs[...,0]/normall,xs[...,1]/normall,c=reshape_joint(joint_out).T.flatten()
                ,cmap="Blues",norm="log",vmin=0.0001,vmax=vmax)

def make_colourbar(fig,cbar_ax,vmax):
    """add color bar to fig"""

    pos = cbar_ax.get_position()

    sm =  ScalarMappable(norm="log",cmap=plt.get_cmap("Blues"))
    sm.set_array([])
    sm.set_clim(vmin=0.001, vmax=vmax)

    cb = fig.colorbar(sm,
                        cax=cbar_ax,
                        orientation="horizontal")

    cbar_ax.set_xlabel("Joint Distribution")
    cbar_ax.set_position([0.25, 0.1, 0.5, 0.02])

def plot_joint_panel(ax,time,jointout,xout,boundary,benchmark=None):
    
    ax, ax_pmarginal,ax_qmarginal = scatter_panel_layout(ax,time,boundary)

    joint_distributions_scatter_panel(ax,
                                            ax_pmarginal,
                                            ax_qmarginal,
                                            jointout,
                                            xout,time,vmax=0.5,benchmark=benchmark)

def contour_plot_(axs,net,boundary,
                            resultsdir,
                            plottimes = [0,1,2,3,4,5],
                            benchmark=None):
    
    
    #axs = make_gridspec_subplots(fig,gs_top)
   
    Q,P = generate_mesh(x_samples=20,qmax=4)
    xinit = torch.stack((Q, P), dim=-1).reshape(-1, 2)
    
    for i in range(len(plottimes)):

        jointout = compute_pdf(xinit.clone(),net,
                                    boundary.sigma0,
                                    boundary.params["epsilon"],
                                    boundary.p_initial,
                                    plottimes[i])
            
        plot_joint_panel(axs[i],plottimes[i],jointout.cpu().numpy(),xinit.clone().cpu().numpy(),
                            boundary = boundary,
                            benchmark = benchmark)
    
    #
    

def make_contour_plot_(net,boundary,
                            resultsdir,
                            plottimes = [0,1,2,3,4,5],
                            benchmark=None):

    fig = plt.figure(figsize=(20, 12), constrained_layout=False)

    # Outer grid: 2 rows (plots + colorbar + legend)
    gs = fig.add_gridspec(nrows=3,
                          ncols=1,
                            left=0, right=1,
                            bottom=0, top=1,
                            wspace=0.3,
                            hspace=0.3,
                          height_ratios=[10, 0.5, 1])                                    

    ncols_top=len(plottimes)//2
    nrows_top=2

    # ---- TOP: nested subplots grid ----
    gs_top = gs[0].subgridspec(nrows_top, ncols_top, wspace=0.05)
    
    axs = make_gridspec_subplots(fig, gs_top, ncols_top,nrows_top)
    contour_plot_(axs,net,boundary,
                            resultsdir,
                            plottimes,
                            benchmark)

    # ---- BOTTOM: colorbar axis ----
    cax = fig.add_subplot(gs[1])
    make_colourbar(fig,cax,vmax=0.5)



    #add legend
    lax = fig.add_subplot(gs[2])
    lax.axis("off")  # hide legend axis box
    h,l = get_all_subplot_handles_labels(fig)
    make_legend(lax,h,l)
    fig.subplots_adjust(hspace=1.1,wspace=1,bottom=0.38)
    save_fig_(fig,"contour_distribution",loc=resultsdir)
    
    plt.close()