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


def perturbative_du_interp(benchmark,mlp_net):
    """Trains a neural network to approximate the perturbatively 
    estimated drift. If a checkpoint is alread in the benchmark solution
    folder, this is loaded instead of training.

    args: 
        benchmark : class
            find the benchmark solution
        mlpnet : class
            neural network
    """

    try:
        load_checkpoint(mlp_net,benchmark.folder_name+f"/perturbative_underdamped_drift.pt")
        print("Checkpoint for perturbative Solution found and loaded")
    
    except BaseException:
        
        #this is any caught error raised by the loadcheckpoint function 
        #It happens when the checkpoint file could not be loaded
        #because checkpoint doesnt exist or is incompatible with initialised network
        
        #train it to learn the csv in the benchmark solution folder
        train_and_save_perturbative_du(benchmark,mlp_net)
    
    return 

def train_and_save_perturbative_du(benchmark,mlp_net):
    run_pretrain(mlp_net,benchmark)
    save_nn_checkpoint(mlp_net,loc=benchmark.get_benchmark_path()+f"/perturbative_underdamped_drift.pt")
    

def contour(plotter,net):
    """Plots a contour plot of the solution to the Fokker-Planck 
    driven by the perturbatively estimated potential. 
    args: 
        boundary : class
            used for parsing into the plotting and integrator classes
        net : class
            trained neural net estimating a drift. 
    """
    
    integrator1 = EulerMaruyamaSDE(boundary,net=net)

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
    
if __name__=="__main__":
    from ..benchmark import *
    from ...sde.boundary import *
    
    boundary = UnderdampedLandauer()

    #get the benchmark solution folder
    benchmark = PerturbativeBenchmarkSolution(boundary,folder="results_temp2")
    
    mlp_net = perturbative_du_interp(benchmark)
    contour(boundary,benchmark,mlp_net)