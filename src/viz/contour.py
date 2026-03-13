import numpy as np 
import matplotlib.pyplot as plt 
import torch

from .plots import *
from .checkpoint_plots import PlotUtils,compute_rho_girsanov

##creates the contour plot showing the full joint pdf

def get_norm(x,y):
    idx = np.argsort(x)
    return np.trapezoid(y[idx],x[idx])

def add_boundary_conditions_scatter_panel(ax_pmarginal,ax_qmarginal,time,boundary):

    if time==0:
        ax_qmarginal.fill_between(
                boundary.q_marginal_initial(np.linspace(-boundary.qmax,boundary.qmax,)),np.linspace(-boundary.qmax,boundary.qmax,),color="orange",label="Assigned Initial Distribution")
        
        ax_pmarginal.fill_between(np.linspace(-boundary.qmax,boundary.qmax,),
                normal_distribution(np.linspace(-boundary.qmax,boundary.qmax,),0,1),color="orange",label="Assigned Initial Distribution")

    elif time==boundary.params["Tf"]:
        ax_qmarginal.fill_between(
                boundary.q_marginal_final(np.linspace(-boundary.qmax,boundary.qmax,)),np.linspace(-boundary.qmax,boundary.qmax,),color="orange",label="Target Final Distribution")
        
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
    qmarg = np.trapezoid(joint_out_temp,x = p_init,axis=0)#np.sum(joint_out_temp,axis=0) #np.trapezoid(joint_out_temp,x = p_init,axis=0)
    pmarg = np.trapezoid(joint_out_temp,x = p_init,axis=1)

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
        x,ptx = benchmark.get_quantity(np.round(time,6),quantity = "underdamped marginal",data="perturbative underdamped",time_type="t0")
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


def make_contour_plot(plotutils):
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

def make_contour_plot2(plotutils,plottimes = [0,1,2,3,4,5]):
    
    ncols = int(np.ceil(len(plottimes)/2))
    fig,axs = make_fig_subplots(ncols=ncols,nrows=2)
    axs = axs.flatten()

    for i in range(len(plottimes)):

        xout,jointout = compute_rho_girsanov(integrator=plotutils.integrator,x_samples=10,mc_samples=plotutils.mc_samples,Tf=plottimes[i])

        plot_joint_panel(axs[i],plottimes[i],jointout,xout,boundary= plotutils.boundary_cond,benchmark=plotutils.referencesol)
    
    fig.subplots_adjust(hspace=1.1,wspace=1,bottom=0.08)
    
    #add legend
    h,l = get_all_subplot_handles_labels(fig)
    make_legend(fig,h,l)
    make_colourbar(fig,vmax=0.5)
    
    plt.tight_layout()

    save_fig_(fig,"contour_distribution",loc=plotutils.resultsdir)
    
    plt.close()


if __name__=="__main__":

    plot_utils = PlotUtils()
    plot_utils.load_du_checkpoint()
    make_contour_plot2(plot_utils)
