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
    Q,P = generate_mesh(x_samples,qmax=5)
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

    #if time == None:
    #    time = boundary.params["Tf"]

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
    r"""Compute normalised q and p marginals from joint distribution data
    args:
        xs : ndarray
        joint_out : ndarray
        see joint_distributions_scatter_panel
    returns: 
        pinit : ndarray
        set of q coordinates
        qmargnorm : ndarray
        normalised position marginal
        pmargnorm : ndarray
        normalised momentum marginal


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
    Plots one panel of the joint distribution, where the 
    center shows the joint and marginals are estimated

    args:
        ax: artist
        axes where to plot. can be a gridspec subplot
        joint_out: arr
        joint distribution (n**2,1) as a numpy array
        xs: arr
        2d input data shaped (n**2,2) as numpy arr (q,p). Assume that the input values of x are generated
        using meshgrid, and so have the same values of q and p
        vmax: float
        maximum value for the scatter plots color scale
    """

    p_init_out,qmarg_norm,pmarg_norm,normall = compute_marginals_from_joint(xs,joint_out)

    #plot and format marginals
    plot_marginal_joint(ax_qmarginal,p_init_out,qmarg_norm)
    plot_marginal_joint(ax_pmarginal,pmarg_norm,p_init_out)

    if benchmark is not None:
        #plot marginals
        x,ptx = benchmark.get_benchmark_quantity(np.round(time,6),quantity = "underdamped marginal",data="perturbative underdamped",time_type="t0")
        ax_qmarginal.plot(ptx,x,color="gray",linestyle="dashed",label="Benchmark Evolved")

    #plot joint distribution
    ax.scatter(xs[...,0]/normall,xs[...,1]/normall,c=reshape_joint(joint_out).T.flatten()
                ,cmap="Blues",norm="log",vmin=0.0001,vmax=vmax)

def make_colourbar(fig,vmax):
    #adds a colour bar

    sm =  ScalarMappable(norm="log",cmap=plt.get_cmap("Blues"))
    sm.set_array([])
    sm.set_clim(vmin=0.001, vmax=vmax)

    cbar_ax = fig.add_axes([0.2, -0.15, 0.55, 0.05])
    cb = fig.colorbar(sm,
                        cax=cbar_ax,
                        orientation="horizontal")

    cb.ax.set_xlabel("Joint Distribution")

def plot_joint_panel(ax,time,jointout,xout,boundary,benchmark=None):
    ax, ax_pmarginal,ax_qmarginal = scatter_panel_layout(ax,time,boundary)

    joint_distributions_scatter_panel(ax,
                                            ax_pmarginal,
                                            ax_qmarginal,
                                            jointout,
                                            xout,time,vmax=0.5,benchmark=benchmark)


def make_contour_plot():
    """plot the learned distribution at final time"""
    
    xout,jointout = plotutils.compute_rho_nn(x_samples=20)
    
    fig,ax = plt.subplots()
    ax, ax_pmarginal,ax_qmarginal = scatter_panel_layout(ax)

    joint_distributions_scatter_panel(ax,
                                                ax_pmarginal,
                                                ax_qmarginal,
                                                jointout,
                                                xout,
                                                time=plotutils.params["Tf"],vmax=0.5,benchmark=plotutils.benchmark)
    
    h,l = get_all_subplot_handles_labels(fig)
    make_legend(fig,h,l)
    make_colourbar(fig,vmax=0.5)
    
    save_fig_(fig,"contour_distribution",loc=plotutils.resultsdir)
    
    plt.close()


def make_contour_plot_(net,boundary,
                            resultsdir,
                            plottimes = [0,1,2,3,4,5],
                            benchmark=None):
    
    ncols = int(np.ceil(len(plottimes)/2))
    fig,axs = make_fig_subplots(ncols=ncols,nrows=2)
    axs = axs.flatten()
    Q,P = generate_mesh(x_samples=10,qmax=5)
    xinit = torch.stack((Q, P), dim=-1).reshape(-1, 2)
    
    for i in range(len(plottimes)):

        jointout = compute_pdf(xinit.clone(),net,
                                    boundary.sigma0,
                                    boundary.params["epsilon"],
                                    boundary.p_initial,
                                    plottimes[i],
                                    mc_samples =2000,
                                    tsteps=10)
            
        plot_joint_panel(axs[i],plottimes[i],jointout.cpu().numpy(),xinit.clone().cpu().numpy(),
                            boundary= boundary,
                            benchmark=benchmark)
    
    fig.subplots_adjust(hspace=1.1,wspace=1,bottom=0.08)
    
    #add legend
    h,l = get_all_subplot_handles_labels(fig)
    make_legend(fig,h,l)
    make_colourbar(fig,vmax=0.5)
    
    save_fig_(fig,"contour_distribution",loc=resultsdir)
    
    plt.close()
