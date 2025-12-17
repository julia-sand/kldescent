"""
trains a simple neural network to interpolate the estimated
perturbative drift from the csv
Computes the solution of the fokker planck and plots
contour plots
"""

import torch
import matplotlib.pyplot as plt 

from ...model.nn import MLP
from ...model.pretrain import run_pretrain
#
from ...sde.langevin import *
from  .marginalhalf import MarginalHalfBridgePerturbation

def perturbative_du_interp(reference_sol):
    """Trains a neural network to approximate the perturbatively 
    estimated drift. If a checkpoint is included in the reference solution
    folder, this is loaded instead of training.

    args: 
        boundary : class
            class for specifying the boundary conditions and hence the reference solution
        reference_sol : class
            controls how the reference solution is saved
    """

    #initialise a simple neural net
    mlp_net = MLP()

    try: #load checkpoint if available
        mlp_net.load_state_dict(torch.load(reference_sol.get_ref_path()+f"/perturbative_underdamped_drift.pt", weights_only=True))
        print("Perturbative Solution found and loaded")
    except FileNotFoundError: 
        print("No perturbative solution found. Training from CSV data.")
        #train it to learn the csv in the reference solution folder
        #try: 
        run_pretrain(mlp_net,reference_sol)
        #except FileNotFoundError:
        #    #first compute perturbative data 
        #    run_pretrain(mlp_net,reference_sol)

        #save the trained checkpoint
        torch.save(mlp_net.state_dict(),reference_sol.get_ref_path()+f"/perturbative_underdamped_drift.pt") 

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

    #plotter = PlotFuns(boundary,referencesol=reference_sol)
    integrator1 = SDE(boundary,net=net)

    times = [0,2.5,5]

    fig,axs = plt.subplots(ncols=len(times),nrows=1)
    fig.set_figheight(15)
    fig.set_figwidth(25)
    for ax in enumerate(axs):
        xout,jointout = compute_rho_girsanov(net,integrator1,mc_samples=1000,Tf=times[ax[0]])

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
    reference_sol = RefSolution(boundary)
    
    mlp_net = perturbative_du_interp(reference_sol)
    contour(boundary,reference_sol,mlp_net)