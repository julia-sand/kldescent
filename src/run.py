import os

import numpy as np
import torch
from torch.nn import DataParallel
from tqdm import tqdm

from src.model import *
from src.sde import *
import src.sde.boundary as boundaries
import src.sde.terminalcost as terminalcosts
from src.utils import *
#import src.viz.plots as plots
from src.viz.checkpoint_plots import *
import src.run_pretrain as pretrain

def run():

    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()
    #overwrite the boundary conditions
    metrics_logger.log_dict(args)

    #initialise the model using parsed boundary conditions
    boundaryconditions = getattr(boundaries,args["boundary"])(logger=metrics_logger)
    
    net = nn.NeuralNetwork(boundaryconditions).to(args["device"])

    #log the params
    metrics_logger.log_dict(net.params)

    #use data paralisation to speed up evaluations
    net = DataParallel(net)
   
    terminal_cost = getattr(terminalcosts,args["terminalcost"])(boundaryconditions)
    
    #intialise sde integration 
    integrator = langevin.SDE(boundaryconditions, net=net, terminal_cost=terminal_cost)
    
    #initialise integration of fpk
    metrics_logger.log_dict(integrator.sde_params)
    metrics_logger.log_dict(terminal_cost.rhonet.module.params)
    
    #initialise trainer & ref solution computation
    trainer = train.Train(net,integrator) 

    metrics = train.Metrics(terminal_cost.rhonet, net, integrator, args["logstep"],metrics_logger)    
        
    if args["pretrain"]: #if a path is provided 
        pretrain.run_pretrain(net,
                            args["pretrain"])
    
    #get baseline metrics and saves initialisation checkpoint
    metrics.update_metrics(0, 0)
    
    #compute reference metrics for final value function calculation
    #integrator.update_ref_statistics()

    for epoch in tqdm(range(args["epochs"])):#range(0,args["epochs"]):
        try:
            #update the terminal cost with current final distribution
            terminal_cost.update_cost_params(integrator)

            #update the drift nn with the current learned final distribution  
            loss_out = trainer.train_step()

            #update metrics
            metrics.update_metrics(epoch+1, loss_out)
            
        except ValueError:
            print("The training has resulted in NaN loss and therefore failed. Please retry.")
            raise
        
    plot_utils = PlotUtils(metrics_logger.get_results_path(),args["epochs"])

    plot_utils.checkpoint_plots_rho()
    plot_utils.checkpoint_plots_du()


if __name__=="__main__":
    run()
