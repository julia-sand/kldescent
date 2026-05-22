import torch

import wandb
import psutil

from kldescent.sde.kl import static_kl_divergence
from kldescent.utils.parser import metric_params
from kldescent.benchmark.benchmarksol import PerturbativeBenchmarkSolution,GaussBenchmarkSolution

def log_cpu(wandb_run):
    #log cpu usage statistics
    wandb_run.log({
        "cpu_percent": psutil.cpu_percent(),
        "num_threads": psutil.Process().num_threads(),
        "ram_percent": psutil.virtual_memory().percent
    })

class Metrics():
    """
    Class for finding metrics during the training process 

    Method:
        update_metrics
    """

    def __init__(self, net, integrator, metrics_logger, bridge_type):
        """
        Initialize the solver with its core components.

        Parameters
        ----------
        net : object
            Model or neural network used in the computation.
        integrator : object
            Object responsible for performing time integration.
        metrics_logger : object
            Utility class for saving the recorded metrics during execution.
        bridge_type : str
            Type of bridge formulation to use (e.g., "marginal", "half").
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
                self.benchmark_.initialise_net(self.benchmark_.bridge_type)
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

        self.kldyn = 0
        self.lr = 0 

    def update_metrics(self, epoch, loss_out):
        """
        Computes training metrics and checks for NaN loss values.

        Parameters
        ----------
        epoch : int
            The current epoch number, used for logging.
        loss_out : float
            The loss value returned by the metrics function.

        Returns
        -------
        None
        """
        with torch.no_grad():

            kl_dynamic,xinit,xref = self.integrator.compute_metrics(self.benchmark_)

            kl_static_from_ref,kl_static = static_kl_divergence(xinit,xref,self.integrator.boundary)
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
