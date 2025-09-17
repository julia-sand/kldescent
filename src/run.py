import os

import numpy as np
import torch
from torch.nn import DataParallel

from src.model import *
from src.sde import *
import src.sde.boundary as boundaries
from src.utils import *
#import src.viz.plots as plots
from src.viz.checkpoint_plots import *
import src.run_pretrain as pretrain

def run():

    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()
    metrics_logger.log_dict(args)

    #initialise the model
    boundaryconditions = getattr(boundaries,args["boundary"])(logger=metrics_logger)
    
    net = nn.NeuralNetwork(boundaryconditions).to(args["device"])
    rhonet = nn.RhoNeuralNetwork()

    #log the params
    metrics_logger.log_dict(net.params)
    metrics_logger.log_dict(rhonet.params)
    
 
    #use data paralisation to speed up evaluations
    net = DataParallel(net)
    rhonet = DataParallel(rhonet)

    #intialise sde integration 
    integrator = langevin.SDE(boundaryconditions, rhonet)
    #initialise integration of fpk
    fpk_integrator = fpk.FokkerPlanck(boundaryconditions, net)
    
    #initialise trainer & ref solution computation
    trainer = train.Train(net,integrator) 
    metrics = train.Metrics(rhonet, net, integrator, args["logstep"])   
    rhotrainer = train.TrainRho(rhonet,fpk_integrator)
    
    #save initialisations 
    metrics_logger.log_dict(integrator.sde_params)


    if args["pretrain"]: #if a path is provided 
        pretrain.run_pretrain(net,
                            args["pretrain"])
    
    #get baseline metrics and saves initialisation checkpoint
    metrics.update_metrics(0,None,metrics_logger)

    for epoch in range(0,args["epochs"]):
        try:

            #train the rho neural network
            rhotrainer.train_loop(metrics_logger)

            #update the drift nn with the current learned final distribution  
            loss_out = trainer.train_step()

            #update metrics
            metrics.update_metrics(epoch+1,loss_out,metrics_logger)
            print(f"Epoch {epoch+1} complete")
            
        except ValueError:
            print("The training has resulted in NaN loss and therefore failed. Please retry.")
            raise
        
    plot_utils = PlotUtils(metrics_logger.get_results_path(),args["epochs"]-1)

    plot_utils.checkpoint_plots_rho()
    plot_utils.checkpoint_plots_du()


if __name__=="__main__":
    run()
