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
from ...sde.boundary import *
from ...sde.langevin import *
from ..reference import *
from ...viz.checkpoint_plots import compute_rho_girsanov
from ...viz.plots import PlotFuns


def perturbative_du_interp(boundary):

    #get the reference solution 
    reference_sol = RefSolution(boundary)
       
    #initialise a neural net
    mlp_net = MLP()


    try: #load checkpoint if available
        mlp_net.load_state_dict(torch.load(reference_sol.get_ref_path()+f"/perturbative_underdamped_drift.pt", weights_only=True))
        print("Checkpoint found and loaded")
    except FileNotFoundError: 
        print("No checkpoint found. Training from CSV data.")
        #train it to learn the csv
        run_pretrain(mlp_net,reference_sol.perturb_underdamped_loc)
        torch.save(mlp_net.state_dict(),reference_sol.get_ref_path()+f"/perturbative_underdamped_drift.pt") 

    return mlp_net

def contour(boundary,net):
    plotter = PlotFuns(boundary)
    integrator1 = SDE(boundary,net=net)

    times = [0,2.5,5]

    fig,axs = plt.subplots(ncols=len(times),nrows=1)
    fig.set_figheight(15)
    fig.set_figwidth(25)
    for ax in enumerate(axs):
        xout,jointout = compute_rho_girsanov(net,integrator1,mc_samples=100,Tf=times[ax[0]])

        plotter.plot_joint_panel(ax[1],times[ax[0]],jointout,xout)
    fig.tight_layout()
    fig.savefig("test_contour_distribution.png",bbox_inches="tight")        

    plt.close()
    plotter.plot_drift(times = [0,0.1,0.2],net=net)
    
if __name__=="__main__":
    boundary = UnderdampedLandauer()
    mlp_net = perturbative_du_interp(boundary)
    contour(boundary,mlp_net)