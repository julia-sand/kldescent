import os

import numpy as np
import torch
from tqdm import tqdm
import random

from kldescent.model import *
from kldescent.sde import *
import kldescent.sde.boundary as boundaries
from kldescent.utils import *
from kldescent.viz.loss_plots import plot_loss_metrics
import kldescent.model.pretrain as pretrain
from kldescent.trainloop import train_loop_du

#set seeds
random.seed(parser.seed())
np.random.seed(parser.seed())
torch.manual_seed(parser.seed())

def run():
    
    print(f"{torch.get_num_threads()} threads and {torch.get_num_interop_threads()} interop threads being used by torch")
    
    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()

    if args["bridgetype"]=="marginal":
        import kldescent.sde.terminalcost_marginal as terminalcosts
    elif args["bridgetype"]=="half":        
        import kldescent.sde.terminalcost as terminalcosts
    else: 
        pass

    #initialise the model using parsed boundary conditions
    boundaryconditions = getattr(boundaries,args["boundary"])(logfile=metrics_logger.datayaml)
       
    net = nn.NeuralNetwork(boundaryconditions,args["device"])
    
    terminal_cost = getattr(terminalcosts,args["terminalcost"])(boundaryconditions,fileloc=metrics_logger.newpath,filename=metrics_logger.datayaml)
    
    #intialise sde integration 
    integrator = langevin.EulerMaruyamaSDE(boundaryconditions, net=net, terminal_cost=terminal_cost)
    
    #initialise trainer
    trainer = train.Train(net,integrator) 

    metric_ = metrics.Metrics(net,integrator, metrics_logger,args["bridgetype"])    

    #save the parameters
    for params in [args,net.params,integrator.sde_params]:
        logger.save_params(params,metrics_logger.datayaml)
    
    if args["pretrain"] is not None: #if a path is provided 
        pretrain.run_pretrain(net,
                            path_to_pretrain=args["pretrain"])
    
    for epoch in tqdm(range(args["epochs"])):#range(0,args["epochs"]):
        
        train_loop_du(epoch,
                        metric_,
                        terminal_cost,
                        integrator,
                        trainer,
                        logstep=args["logstep"])
    
    if metric_.wandb_run is not None:
        metric_.wandb_run.finish()

    plot_loss_metrics(folder=metrics_logger.newpath)

if __name__=="__main__":
    run()
