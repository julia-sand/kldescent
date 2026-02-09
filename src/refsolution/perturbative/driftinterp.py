"""
Trains a simple neural network to interpolate the estimated
perturbative drift from the csv

When called directly also computes the solution of the fokker planck and plots
contour plots
"""

import torch

import matplotlib.pyplot as plt 

from ...model.nn import save_nn_checkpoint,load_checkpoint
from ...model.pretrain import run_pretrain


def perturbative_du_interp(reference_sol,mlp_net):
    """Trains a neural network to approximate the perturbatively 
    estimated drift. If a checkpoint is included in the reference solution
    folder, this is loaded instead of training.

    args: 
        boundary : class
            class for specifying the boundary conditions and hence the reference solution
        reference_sol : class
            controls how the reference solution is saved
    """

    try:
        #mlp_net.load_state_dict(torch.load(reference_sol.get_ref_path()+f"/perturbative_underdamped_drift.pt", weights_only=True))
        load_checkpoint(mlp_net,reference_sol.get_ref_path()+f"/perturbative_underdamped_drift.pt")
        print("Checkpoint for perturbative Solution found and loaded")
    except BaseException:
        #train it to learn the csv in the reference solution folder
        run_pretrain(mlp_net,reference_sol)

        #save the trained checkpoint
        #torch.save(mlp_net.state_dict(),reference_sol.get_ref_path()+f"/perturbative_underdamped_drift.pt") 
        save_nn_checkpoint(mlp_net,loc=reference_sol.get_ref_path()+f"/perturbative_underdamped_drift.pt")


    return mlp_net

def contour(plotter,net):
    """Plots a contour plot of the solution to the Fokker-Planck 
    driven by the perturbatively estimated potential. 
    args: 
        boundary : class
            used for parsing into the plotting and integrator classes
        net : class
            trained neural net estimating a drift. 
    """
    
    #from ...sde.langevin import *

    #plotter = PlotFuns(boundary,referencesol=reference_sol)
    integrator1 = SDE(boundary,net=net)

    times = [0,2.5,5]

    fig,axs = plt.subplots(ncols=len(times),nrows=1)
    fig.set_figheight(15)
    fig.set_figwidth(25)
    for ax in enumerate(axs):
        xout,jointout = compute_rho_girsanov(integrator=integrator1,mc_samples=1000,Tf=times[ax[0]])

        plotter.plot_joint_panel(ax[1],times[ax[0]],jointout,xout)
    
    fig.tight_layout()
    fig.savefig("test_contour_distribution.png",bbox_inches="tight")        

    plt.close()
    plotter.plot_drift(times = [0,2.5,5],net=net)
    
if __name__=="__main__":
    from ..reference import *
    from ...sde.boundary import *
    
    boundary = UnderdampedLandauer()

    #get the reference solution folder
    reference_sol = RefSolution(boundary,folder="results_temp2")
    
    mlp_net = perturbative_du_interp(reference_sol)
    contour(boundary,reference_sol,mlp_net)