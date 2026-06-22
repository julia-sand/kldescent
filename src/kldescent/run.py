import numpy as np
import torch
from tqdm import tqdm
import random

from kldescent.model import *
from kldescent.sde import *
import kldescent.sde.boundary as boundaries
import kldescent.sde.loss as loss_fns
from kldescent.utils import *
from kldescent.viz.loss_plots import plot_loss_metrics
from kldescent.model.pretrain import run_pretrain
from kldescent.model.nn import save_nn_checkpoint
from kldescent.trainloop import train_loop_du


def run():
    

    #set seeds
    s = parser.seed()    
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    #print(f"{torch.get_num_threads()} threads and {torch.get_num_interop_threads()} interop threads being used by torch")
    
    #create a log file and results directory
    metrics_logger = logger.LossLogger()

    #parse top-level arguments from command line
    args = parser.overdamped_argparser()

    #initialise the model using parsed boundary conditions
    boundaryconditions = getattr(boundaries,args["boundary"])(logfile=metrics_logger.datayaml)
       
    #initialise class for evaluating terminal cost, depending on bridge type
    if args["bridgetype"]=="marginal":
        import kldescent.sde.terminalcost_marginal as terminalcosts
    elif args["bridgetype"]=="half":        
        import kldescent.sde.terminalcost as terminalcosts
    else: 
        raise NotImplementedError

    terminal_cost = getattr(terminalcosts,args["terminalcost"])(boundaryconditions,fileloc=metrics_logger.newpath,filename=metrics_logger.datayaml)
    
    #initialise neural network and compile model
    net = nn.NeuralNetwork(boundaryconditions,args["device"])
    #net = torch.compile(net, backend="eager")

    #intialise sde integration method
    integrator = getattr(langevin, args["integrator"])(boundaryconditions, net=net)

    #initialise loss function
    _loss_cls = {"kl": loss_fns.KL, "ep": loss_fns.ep}[args["loss"]]
    loss = _loss_cls(integrator, terminal_cost)

    #initialise trainer
    trainer = train.Train(net, loss)

    #initialise metrics and action cost computations
    metric_ = metrics.Metrics(net,integrator, metrics_logger,args["bridgetype"])    

    #log the parameters
    for params in [args,net.params,integrator.sde_params]:
        logger.save_params(params,metrics_logger.datayaml)
    
    #pretrain the network with data from csv if provided
    if args["pretrain"] is not None: #if a path is provided 
        run_pretrain(net,
                    metrics_logger.newpath,
                    path_to_pretrain=args["pretrain"],
                    epsilon=None)
                                          
    #save initial checkpoint before any gradient updates
    save_nn_checkpoint(net, loc=metrics_logger.newpath+f"/checkpoints/{net.net_name}_checkpoint_0.pt")

    #training
    for epoch in tqdm(range(1, args["epochs"] + 1)):
        
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
