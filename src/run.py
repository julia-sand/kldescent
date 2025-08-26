import os

import numpy as np
import torch
from torch.nn import DataParallel

from src.model import *
from src.sde import *
import src.sde.boundary as boundaries
from src.utils import *
import src.viz.plots as plots
from src.viz.checkpoint_plots import *

def run():

    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()
    metrics_logger.log_dict(args)

    #initialise the model
    boundaryconditions = getattr(boundaries,args["boundary"])(logger=metrics_logger)
    if args["boundary"]=="LinearOU":
        net = nn.MLP().to(args["device"])
    else:
        net = nn.NeuralNetwork(boundaryconditions).to(args["device"])

    rhonet = nn.RhoNeuralNetwork(boundaryconditions)
    

    #log the params
    metrics_logger.log_dict(net.params)
    metrics_logger.log_dict(rhonet.params)
    
    #use data paralisation to speed up evaluations
    net = DataParallel(net)
    rhonet = DataParallel(rhonet)

    if args["pretrain"]: #TODO this needs updating
        import run_pretrain
        run_pretrain.run_pretrain(net,metrics_logger.get_results_path(),boundaryconditions)
    
    #intialise sde integration 
    integrator = langevin.SDE(boundaryconditions, rhonet)
    #initialise integration of fpk
    fpk_integrator = fpk.FokkerPlanck(boundaryconditions, net)
    
    #save params
    metrics_logger.log_dict(integrator.sde_params)

    #initialise trainer & ref solution computation
    trainer = train.Train(net,integrator) 
    metrics = train.Metrics(trainer,args["logstep"])
    rhotrainer = train.TrainRho(rhonet,fpk_integrator)
    
    #save zero metrics 
    #metrics_logger.save_checkpoint(rhonet,0)
    #metrics.update_metrics(0,0,metrics_logger) #write 

    for epoch in range(1,args["epochs"]+1):
        try:

            #train the rho nn
            rhotrainer.train_loop()
            metrics_logger.save_checkpoint(rhonet,epoch)
  
            #update the drift nn with the current iteration  
            loss_out = trainer.train_step()
            #update metrics
            metrics.update_metrics(epoch,loss_out,metrics_logger)
        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {epoch}")
        except ValueError:
            print("The training has resulted in NaN loss and therefore failed. Please retry.")
            raise
        except slurmhandler.TimeLimitException: #TODO not sure if this works?
            print(f"The training must be stopped because SLURM is about to kill the job. Training ended at epoch {epoch}")
            metrics_logger.save_checkpoint(net,epoch)


    #plotter = plots.PlotFuns(boundaryconditions)
    #plotter.setresultsdir(metrics_logger.get_results_path())
    #checkpoint_plots_all(metrics_logger,
    #                        boundaryconditions,
    #                        net,
    #                        plotter,
    #                        integrator)
    #plotter.plot_loss_metrics()

if __name__=="__main__":
    run()
