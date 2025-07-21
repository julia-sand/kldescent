import os
import yaml 

import torch
import numpy as np

from ..model.nn import NeuralNetwork
from .plots import PlotFuns
from ..sde import langevin
from ..utils.parser import checkpoint_plot_params,overdamped_argparser, fetch_results_folder_from_cmd
from ..sde import boundary as boundaries 

def load_net(boundary_cond,epoch,param_data,folder=None):

    if folder == None:
        folder = "tests/"+f"{boundary_cond.__class__.__name__}"
    else: #if a folder path is provided, assume that the reference solution is there and overwrite the default    
        boundary_cond.overwrite_ref_path(folder)

    #print(folder+f"/checkpoints/net_checkpoint_{epoch}.pt")

    net = NeuralNetwork(boundary_cond,param_data)
    #net = DataParallel(net)

    net.load_checkpoint(folder+f"/checkpoints/net_checkpoint_{epoch}.pt")


    plotter = PlotFuns(boundary_cond)
    #make new folder for the results
    os.makedirs(folder+f"/plots_{epoch}", exist_ok = True) 
    plotter.setresultsdir(folder+f"/plots_{epoch}")

    integrator = langevin.SDE(boundary_cond,param_data)

    return net, plotter, integrator


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

    folder = fetch_results_folder_from_cmd()
        
    try:
        # Read YAML file
        with open(f'{folder}/data.yaml', 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        boundary_cond = getattr(boundaries,data_loaded["boundary"])(params=data_loaded)
    except FileNotFoundError: #if no yaml file is provided, use the defaults
        print("No parameter file given. The plots will be created using the default values. See -h for values.")

        data_loaded = None
        
        run_params = overdamped_argparser()
        boundary_cond = getattr(boundaries,run_params["boundary"])()


    params = checkpoint_plot_params() #get checkpoint plot parameters from cmd line
    
    net, plotter, integrator = load_net(boundary_cond,params["epoch"],data_loaded,
                                        folder)    
    integrator.compute_sde(net,
                            plotter,
                            boundary_cond.params["Tf"])
    plotter.plot_drift(times = [0,boundary_cond.params["Tf"]/2,boundary_cond.params["Tf"]],net=net)
    #print(compute_ref_kl(boundary_cond))
