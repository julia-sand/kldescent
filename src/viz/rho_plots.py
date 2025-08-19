"""
Loads checkpoint file (if provided) for neural network modelling the probability 
distribution and makes some plots 
"""

import os
import yaml 

import torch
import numpy as np

from ..model.nn import NeuralNetwork, RhoNeuralNetwork
from .plots import PlotFuns
from ..sde import langevin
from ..utils.parser import checkpoint_plot_params,fetch_results_folder_from_cmd
from ..utils.getboundary import *

class PlotUtils():
    """
    class which controls the 
    """

    def __init__(self,boundary_cond,epoch,param_data,folder=None):
        self.net = NeuralNetwork(boundary_cond,param_data)
        self.rhonet = RhoNeuralNetwork(boundary_cond,param_data)
        
        self.plotter = PlotFuns(boundary_cond)

        if folder == None:
            folder = "tests/"+f"{boundary_cond.__class__.__name__}"
        else: #if a folder path is provided, assume that the reference solution is there and overwrite the default    
            boundary_cond.overwrite_ref_path(folder)
            self.net.load_checkpoint(folder+f"/checkpoints/{self.net.net_name}_checkpoint_{epoch}.pt")
            self.rhonet.load_checkpoint(folder+f"/checkpoints/{self.rhonet.net_name}_checkpoint_{epoch}.pt")

        #make new folder for the results
        os.makedirs(folder+f"/plots_{epoch}", exist_ok = True) 
        self.plotter.setresultsdir(folder+f"/plots_{epoch}")

        self.integrator = langevin.SDE(boundary_cond,sde_params=param_data)


def checkpoint_plots_all(logger,
                            boundary_cond,
                            net,
                            plotter,
                            integrator):

    folder = logger.get_results_path()
    integrator.compute_sde(net,
                    plotter,
                    boundary_cond.params["Tf"])
    plotter.plot_drift(times = [0,boundary_cond.params["Tf"]/2,boundary_cond.params["Tf"]],net=net)

def compute_ref_kl(boundary):
    times = boundary.get_reference_timegrid()
    klout = np.empty(len(times))
    #mean_t = boundary.get_known_solution_cumulants("Position Mean")
    var_t = boundary.get_known_solution_cumulants("Position Variance")
    for t in enumerate(times):
        xref,duref = boundary.get_reference_data(t[1])
        ptx = plotter.normal_distribution(xref,0,var_t[t[0]])
        #np.exp(-(xref-mean_t[t[0]])**2/(2*var_t[t[0]]))/(np.sqrt(2*np.pi*var_t[t[0]]))
        klout[t[0]] = np.trapz(duref*ptx,xref)
    return np.trapz(klout,times)

if __name__=="__main__":

    #get folder and load boundary data and params from file
    folder = fetch_results_folder_from_cmd()   
    boundary_cond,data_loaded = init_params_from_file(folder)

    params = checkpoint_plot_params() #get checkpoint plot parameters from cmd line
    
    plot_setup = PlotUtils(boundary_cond,params["epoch"],
                                        data_loaded,
                                        folder)
    plot_setup.integrator.compute_sde(plot_setup.net,
                            plot_setup.plotter,
                            boundary_cond.params["Tf"])
    plot_setup.plotter.plot_drift(times = [0,boundary_cond.params["Tf"]/2,boundary_cond.params["Tf"]],net=plot_setup.net)
    #print(compute_ref_kl(boundary_cond))
