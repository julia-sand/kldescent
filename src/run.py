import os

import numpy as np
import torch
from tqdm import tqdm

from src.model import *
from src.sde import *
import src.sde.boundary as boundaries
from src.utils import *
from src.viz.loss_plots import plot_loss_metrics
import src.model.pretrain as pretrain
from src.refsolution.reference import *
from src.trainloop import train_loop_du

def run():
    
    print(f"{torch.get_num_threads()} threads and {torch.get_num_interop_threads()} interop threads being used by torch")
    
    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse arguments from command line
    args = parser.overdamped_argparser()

    if args["bridgetype"]=="marginal":
        import src.sde.terminalcost_marginal as terminalcosts
    else:        
        import src.sde.terminalcost as terminalcosts

    #initialise the model using parsed boundary conditions
    boundaryconditions = getattr(boundaries,args["boundary"])(logger=metrics_logger)
       
    net = nn.NeuralNetwork(boundaryconditions).to(args["device"])
    
    if "Gauss" in args["boundary"]:
        refsolution_ = GaussRefSolution(bridge_type=args["bridgetype"],boundary=boundaryconditions,folder_name = metrics_logger.get_results_path())
    else:
        refsolution_ = RefSolution(bridge_type=args["bridgetype"],boundary=boundaryconditions,folder_name = metrics_logger.get_results_path())

    terminal_cost = getattr(terminalcosts,args["terminalcost"])(boundaryconditions,metrics_logger)
    
    #intialise sde integration 
    integrator = langevin.EulerMaruyamaSDE(boundaryconditions, net=net, terminal_cost=terminal_cost)
    
    #initialise trainer & ref solution computation
    trainer = train.Train(net,integrator) 

    metric_ = metrics.Metrics(net,integrator, metrics_logger,refsol=refsolution_)    

    #save the parameters
    logger.save_params(metrics_logger,[args,net.params,integrator.sde_params])
    
    if args["pretrain"]: #if a path is provided 
        pretrain.run_pretrain(net,
                            path_to_pretrain=args["pretrain"])
    
    for epoch in tqdm(range(args["epochs"])):#range(0,args["epochs"]):
        
        train_loop_du(epoch,
                        metric_,
                        terminal_cost,
                        integrator,
                        trainer,
                        refsolution_,
                        logstep=args["logstep"])

        
        
        #except ValueError:
            #print("The training has resulted in NaN loss and therefore failed. Please retry.")
            #raise
    
    plot_loss_metrics(folder=metrics_logger.newpath)

if __name__=="__main__":
    run()
