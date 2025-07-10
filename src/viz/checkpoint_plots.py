#import math
import torch
import numpy as np
#from scipy.stats import bootstrap

import pdb

from ..model.nn import NeuralNetwork
from .plots import PlotFuns
#import src.sde.boundary as boundaries
from ..sde import langevin

def load_net(boundary_cond,epoch,param_data,folder=None):

    if folder == None:
        folder = "tests/"+f"{boundary_cond.__class__.__name__}"
    else: #if a folder path is provided, assume that the reference solution is there and overwrite the default    
        boundary_cond.overwrite_ref_path(folder)

    print(folder+f"/checkpoints/net_checkpoint_{epoch}.pt")

    net = NeuralNetwork(boundary_cond,param_data)
    #net = DataParallel(net)

    net.load_checkpoint(folder+f"/checkpoints/net_checkpoint_{epoch}.pt")


    plotter = PlotFuns(boundary_cond)
    plotter.setresultsdir(folder)
    integrator = langevin.SDEOverdamped(boundary_cond,param_data)

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




if __name__=="__main__":
    from ..utils.parser import checkpoint_plot_params,overdamped_argparser, fetch_results_folder_from_cmd
    import src.sde.boundary as boundaries 
    from torch.nn import DataParallel

    import yaml 

    folder = fetch_results_folder_from_cmd()
        
    try:
        # Read YAML file
        with open(f'{folder}/data.yaml', 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        boundary_cond = getattr(boundaries,data_loaded["boundary"])(data_loaded)
    except FileNotFoundError:
        data_loaded = None
        #use defaults
        run_params = overdamped_argparser()
        boundary_cond = getattr(boundaries,run_params["boundary"])()


    params = checkpoint_plot_params()
    
    net, plotter, integrator = load_net(boundary_cond,params["epoch"],data_loaded,
                                        folder)    
    integrator.compute_sde(net,
                            plotter,
                            boundary_cond.params["Tf"])
    plotter.plot_drift(times = [0,boundary_cond.params["Tf"]/2,boundary_cond.params["Tf"]],net=net)
