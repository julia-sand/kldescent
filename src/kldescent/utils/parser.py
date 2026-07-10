"""handles parameter loading from command line"""

import argparse
import os
import yaml

class CustomArgParse(argparse.ArgumentParser):
    """
    Custom argument parser.

    If a ``data.yaml`` file exists inside the ``--folder`` directory, its
    parameter values are used as default parameter values unless given as command-line arguments.

    Without a yaml file the parser falls back to argparse defaults, so
    existing command-line usage is unchanged.
    """

    def __init__(self):
        argparse.ArgumentParser.__init__(self)
        self.allow_abbrev = False
        self.loc = fetch_results_folder_from_cmd()
        self.argfile = self._load_yaml_argfile()

    def _load_yaml_argfile(self):
        """Return a flat arg list from data.yaml, or None if not found."""
        if self.loc is None:
            return None
        yaml_path = os.path.join(self.loc, "data.yaml")
        if not os.path.isfile(yaml_path):
            return None
        
        with open(yaml_path) as f:
            params = yaml.safe_load(f) or {}
        
        args = []
        for key, val in params.items():
            if val is None:
                continue
            args += [f"--{key}", str(val)]
        
        return args



def fetch_results_folder_from_cmd():
    # Instantiate the parser
    parser6 = argparse.ArgumentParser()
    
    parser6.add_argument('-folder','--folder', type=str, default="results2",
                        help='Folder where checkpoints and loss metrics will be saved. Provide filepath from current working directory.')

    args6 = parser6.parse_known_args()

    return args6[0].folder


def fetch_checkpoint_path():
    # Instantiate the parser
    parser6 = CustomArgParse()
    
    parser6.add_argument('-checkpointpath','--checkpointpath', type=str, default=None,
                        help='Path for initialisation checkpointcheckpoint to pick up immediately from training loop. Path should be provided from the current working directory.')

    args6 = parser6.parse_known_args()

    return args6[0].checkpointpath

def cell_argparser():
    """Parse arguments for solving the cell problem in benchmark/iteration"""
    
    # Instantiate the parser
    parser = CustomArgParse()

    parser.add_argument('-cellxnum','--cellxnum', type=int, default=1000,
                        help='how many x values to use in the spacial discretisation when computing the cell problem')
    
    parser.add_argument('-celltnum','--celltnum', type=int, default=11,
                        help='how many t values to use in the temporal discretisation when computing the cell problem')
    
    args = parser.parse_known_args(parser.argfile)

    return vars(args[0])



def train_argparser():
    """Parse arguments for solving the cell problem in benchmark/iteration"""
    
    # Instantiate the parser
    parser = CustomArgParse()

    parser.add_argument('-lr','--lr',  type=float, default=0.0005,
                        help='Initial learning rate.')
    
    parser.add_argument('-startdecay','--startdecay', type=int, default=20,
                        help='Number of epochs to wait before starting learning rate scheduling.')
    
    parser.add_argument('-gradclip','--gradclip', type=float, default=0.5,
                        help='size of gradient clipping ')
    
    args = parser.parse_known_args(parser.argfile)

    return vars(args[0])


def overdamped_argparser():
    #general configs for the training process
    # Instantiate the parser
    parser = CustomArgParse()

    parser.add_argument('-bridgetype','--bridgetype', type=str, default="half",
                        help='The type of bridge configuration. The bridge type applies to the loss function (half and marginal only) and to specify the benchmark problem. Valid values are half, marginal or full')
    
    parser.add_argument('-epochs','--epochs', type=int, default=1,
                        help='Total number of training epochs. This is the number of parameter updates for the drift (du) network')
    
    parser.add_argument('-device','--device', type=str, default="cpu",
                        help='Computing device to use for training. If gpu is selected, first checks if available, and defaults to cpu if none is available.')
    
    parser.add_argument('-logstep','--logstep', type=int, default=1,
                        help='How frequency to save DU neural network checkpoint and compute all loss metrics')
    
    parser.add_argument('-boundary','--boundary', type=str, default="UnderdampedLandauer",
                        help='Set of boundary conditions to use. See Sde.Boundary for options')
    
    parser.add_argument('-terminalcost','--terminalcost', type=str, default="GirsanovMC",
                        help='How the terminal cost is evaluated. See Sde.terminalcost for options')

    parser.add_argument('-integrator','--integrator', type=str, default="BAOAB",
                        help='SDE integrator method to use. Options: BAOAB, EulerMaruyamaSDE, VerletSDE')

    parser.add_argument('-loss','--loss', type=str, default="kl",
                        help='Loss function. Options: kl (kullback-liebler), ep (entropy production)')
    
    parser.add_argument('-pretrain','--pretrain', type=str, default=None,
                        help='Path from current working directory to a CSV containing function values to fit to the neural network before training. See model/pretrain for required structure of the csv file.')
             
    args = parser.parse_known_args(parser.argfile)

    return vars(args[0])


def integrator_params():

    # Instantiate the parser
    integrator_params = CustomArgParse()

    integrator_params.add_argument('-batchsamples','--batchsamples', type=int, 
                        default=64,
                        help='Number of samples in the training set (used to evaluate the loss function)')

    integrator_params.add_argument('-tsteps','--tsteps', type=int, default=20,
                        help='number of timesteps in the integration of the SDE and loss function.')

    integrator_params.add_argument('-testsamples','--testsamples', type=int, default=3,
                        help='number of samples for computing loss metrics (eg static KL). May be also used to compute the terminal cost.')

    integrator_params.add_argument('-kramers','--kramers',  type=float, default=0,
                        help='Kramers coupling constant for regularised dynamics.')

    args1 = integrator_params.parse_known_args(integrator_params.argfile)

    return vars(args1[0])


def boundary_params():

    # Instantiate the parser
    boundary_params = CustomArgParse()

    boundary_params.add_argument('-d','--d', type=int, default=1,
                        help='Dimension of position space.')
    
    boundary_params.add_argument('-Tf','--Tf', type=float, default=5,
                        help='final time')

    boundary_params.add_argument('-epsilon','--epsilon', type=float, default=0.2,
                        help='constant')

    boundary_params.add_argument('-weight','--weight', type=float, default=20,
                        help='M, baldovin constant. Encodes the weighting of the terminal vs running cost.')

    args2 = boundary_params.parse_known_args(boundary_params.argfile)

    return vars(args2[0])


def gaussian_params():
    #initial conditions for a gaussian half bridge

    # Instantiate the parser
    params1 = CustomArgParse()

    params1.add_argument('-muinitial','--muinitial', type=float, default=0,
                        help='Initial value of the mean when using Gaussian boundary conditions')

    params1.add_argument('-mufinal','--mufinal', type=float, default=0,
                        help='Target value of the mean when using Gaussian boundary conditions')

    params1.add_argument('-sigmainitial','--sigmainitial', type=float, default=1,
                        help='Initial value of the standard deviation when using Gaussian boundary conditions')

    params1.add_argument('-sigmafinal','--sigmafinal', type=float, default=0.5,
                        help='Target value of the standard deviation when using Gaussian boundary conditions')

    args5 = params1.parse_known_args(params1.argfile)

    return vars(args5[0])

def du_network_params():
    """Parameters for neural network used to model the drift
    """
    # Instantiate the parser
    parser3 = CustomArgParse()
    
    parser3.add_argument('-duhiddenlayerneurons','--duhiddenlayerneurons', type=int, default=16,
                        help='Number of hidden dimensions in each layer of the Neural network model modelling DU')

    args3 = parser3.parse_known_args(parser3.argfile)

    return vars(args3[0])

def rho_network_params():
    """Parameters for neural network used to find the current final distribution
    """
    # Instantiate the parser
    parser3 = CustomArgParse()
    
    parser3.add_argument('-rhohiddenlayerneurons','--rhohiddenlayerneurons', type=int, default=16,
                        help='Number of hidden dimensions in each layer of the Neural network model modelling the terminal distribution')

    args3 = parser3.parse_known_args(parser3.argfile)

    return vars(args3[0])

def girsanov_plot_params():
    """Parameters for making contour plots based on the Girsanov-theorem method of integrating the Fokker-Planck.
    """
    
    # Instantiate the parser
    parser4 = CustomArgParse()
    
    parser4.add_argument('-xsamples','--xsamples', type=int,
                        default=20,
                        help='how many coordinates per axis. Total number of coordinates is xsamples**2')

    parser4.add_argument('-mcsamples','--mcsamples', type=int, default=1000,
                        help='Number of monte carlo trajectories to use per coordinate.')
                      
    args4 = parser4.parse_known_args(parser4.argfile)

    return vars(args4[0])


def checkpoint_plot_params():
    """Parameters for making checkpoint plots 
    """
    
    # Instantiate the parser
    parser5 = CustomArgParse()
    requiredNamed = parser5.add_argument_group('required named arguments')
    requiredNamed.add_argument('-epoch','--epoch', type=int,
                                help='At which epoch checkpoint to plot. A checkpoint file ending in _epoch must be available, otherwise default initialisation is used.')
    
    requiredNamed.add_argument('-mcsamples','--mcsamples', type=int, default=0,
                                help='Overwrite the test samples parameter in the data file to use different number of samples than for training when making plots.')
    
    args5 = parser5.parse_known_args()

    return vars(args5[0])

def metric_params():

    # Instantiate the parser
    metric_params = CustomArgParse()

    metric_params.add_argument('-usewandb','--usewandb', type=str, default=None,
                        help='WandB project name. If provided, logs metrics to that project. Omit to disable WandB.')

    metric_params.add_argument('-usebenchmark','--usebenchmark', type=str, default=None,
                        help='Filepath to location of the benchmark solution. If none, no benchmark will be plotted.')

    args2 = metric_params.parse_known_args(metric_params.argfile)[0]

    return args2.usebenchmark,args2.usewandb


def seed():

    # Instantiate the parser
    seed = CustomArgParse()

    seed.add_argument('-seed','--seed', type=int, default=1,
                        help='random seed')


    return seed.parse_known_args()[0].seed


if __name__=="__main__":
    

    params_test = overdamped_argparser()

    print(params_test)
