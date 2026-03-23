import torch
import numpy as np 

import wandb
import psutil

from ..model.nn import save_nn_checkpoint
from ..sde.kl import static_kl_divergence
from ..utils.parser import metric_params
from ..benchmark.benchmarksol import PerturbativeBenchmarkSolution,GaussBenchmarkSolution

def log_cpu(wandb_run):
    wandb_run.log({
        "cpu_percent": psutil.cpu_percent(),
        "cpu_per_core": psutil.cpu_percent(percpu=True),
        "num_threads": psutil.Process().num_threads(),
    })

class Metrics():
    """
    Class for finding metrics during the training process 

    Method: 
        update_metrics
    """

    def __init__(self, net, integrator, metrics_logger, bridge_type):
        """
        Initialises class parameters
        args:
            net : class

            integrator : 

            metricslogger : 
            use_perturbative_metrics : Bool. 
                Default False. 
                Set to true to use the perturbative solution as the 
        """
        self.integrator = integrator
        self.net = net

        self.metrics_logger = metrics_logger #csv logger

        self.usebenchmark,self.usewandb = metric_params()

        if self.usebenchmark:
            if "Gauss" in integrator.boundary.__class__.__name__:
                self.benchmark_ = GaussBenchmarkSolution(bridge_type=bridge_type,boundary=integrator.boundary,folder_name = metrics_logger.get_results_path())
                self.benchmarknet = None
            else:
                self.benchmark_ = PerturbativeBenchmarkSolution(bridge_type=bridge_type,boundary=integrator.boundary,folder_name = metrics_logger.get_results_path())
                self.benchmarknet = self.benchmark_.du_net_benchmark 
        else: 
            self.benchmark_ = None

        if self.usewandb:
            self.wandb_run =  wandb.init(
                                    # Set the wandb entity where your project will be logged (generally your team name).
                                    entity="julia-sohkanen-university-of-helsinki",
                                    # Set the wandb project where this run will be logged.
                                    project="kl-descent",
                                    )
        else: 
            self.wandb_run = None

        #self.update_metrics(0, 0)
        #self.kldyn = 0
        self.lr = 0 

    def update_metrics(self, epoch, loss_out):
        """
        Function for computing metrics
            - checks for nan loss
        args:
            epoch:
                pass current epoch to use in logging loss
            loss_out:
                value of the loss from the metrics function
        """
        
        with torch.no_grad():

            if loss_out is not None:
                loss_out = loss_out
            else:
                loss_out = 0 #TODO: compute the loss independently here

            with torch.no_grad():

                kl_dynamic,xinit,xref = self.integrator.compute_metrics(self.benchmark_)

                kl_static_from_ref,kl_static = 0,0#static_kl_divergence(xinit,xref,self.integrator.boundary)
                self.actioncost = kl_dynamic + self.integrator.boundary.params["weight"]*kl_static_from_ref
                if self.wandb_run is not None:
                    self.wandb_run.log({"dynamickl": kl_dynamic, 
                                        "statickl": kl_static, 
                                        "action":self.actioncost,
                                        "lr":self.lr})
                    log_cpu(self.wandb_run)

                self.metrics_logger.write_data_row(epoch,
                                            loss_out,
                                            self.actioncost,
                                            #kl_dynamic + self.integrator.boundary.params["weight"]*kl_static_from_ref,
                                            kl_dynamic,
                                            kl_static,
                                            kl_static_from_ref,
                                            self.lr)
                                           # np.nanvar(xinit[...,0]))
