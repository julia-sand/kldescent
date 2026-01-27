import os

import numpy as np
import torch
from tqdm import tqdm

from src.model import *
from src.sde import *
import src.sde.boundary as boundaries
from src.utils import *
from src.viz.checkpoint_plots import *
import src.model.pretrain as pretrain

def run():

    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()

    if args["bridgetype"]=="marginal":
        import src.sde.terminalcost_marginal as terminalcosts
    else:        
        import src.sde.terminalcost as terminalcosts

    #overwrite the boundary conditions
    metrics_logger.log_dict(args)

    #initialise the model using parsed boundary conditions
    boundaryconditions = getattr(boundaries,args["boundary"])(logger=metrics_logger)
        
    net = nn.NeuralNetwork(boundaryconditions).to(args["device"])
    
    #log the params
    metrics_logger.log_dict(net.params)
    
    terminal_cost = getattr(terminalcosts,args["terminalcost"])(boundaryconditions,metrics_logger)
    
    #intialise sde integration 
    integrator = langevin.SDE(boundaryconditions, net=net, terminal_cost=terminal_cost)
    
    #log integration parameters
    metrics_logger.log_dict(integrator.sde_params)
    
    #initialise trainer & ref solution computation
    trainer = train.Train(net,integrator) 

    metric_ = metrics.Metrics(net,integrator, metrics_logger,use_perturbative_metrics=False)    
        
    if args["pretrain"]: #if a path is provided 
        pretrain.run_pretrain(net,
                            path_to_pretrain=args["pretrain"])
    
    for epoch in tqdm(range(args["epochs"])):#range(0,args["epochs"]):
        try:

            #update the terminal cost with current final distribution
            terminal_cost.update_cost_params(integrator,epoch)

            #update the drift nn with the current learned final distribution  
            loss_out = trainer.train_step()
                
            if (epoch-1) % args["logstep"] == 0:

                #update metrics and save checkpoints
                metric_.update_metrics(epoch+1, loss_out)
                                
                #uncomment to make plots
                #!! creating plots will run perturbative calculations
                #plot_utils = PlotUtils(folder = metrics_logger.get_results_path(),epoch=epoch+1)

                #plot_utils.checkpoint_plots_du()
                #plot_utils.make_girsanov_net_plots()
                #plot_utils.make_contour_plot()

        except ValueError:
            print("The training has resulted in NaN loss and therefore failed. Please retry.")
            raise
    
if __name__=="__main__":
    run()
