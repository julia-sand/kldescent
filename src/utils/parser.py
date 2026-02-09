"""handles parameter loading from command line"""

import yaml 

import argparse

class CustomArgParse(argparse.ArgumentParser):
    """
    Custom argument parser. 
    Sets the parser to read from a file if one is provided, otherwise returns the 
    defaults. 
    """

    #overwrites the default behaviour if using results folder to use the parameter file available
    def __init__(self):
        self.loc = fetch_results_folder_from_cmd()
        if self.loc is not None:
            argparse.ArgumentParser.__init__(self,fromfile_prefix_chars = '@')

            #uncomment this to use a custom file format
            #self.argfile = [f'@{self.loc}/data.txt']

        else:
            argparse.ArgumentParser.__init__(self)
        
        self.argfile = None

        #disallow abbreviations    
        self.allow_abbrev = False

    #TODO update to handle the data.txt format 
    #def convert_arg_line_to_args(self,arg_line):
    #    return arg_line.split


def fetch_results_folder_from_cmd():
    # Instantiate the parser
    parser6 = argparse.ArgumentParser()
    
    parser6.add_argument('-folder','--folder', type=str, default="results2",
                        help='Folder where to run the plots.')

    args6 = parser6.parse_known_args()

    return args6[0].folder


def fetch_checkpoint_path():
    # Instantiate the parser
    parser6 = argparse.ArgumentParser()
    
    parser6.add_argument('-checkpointpath','--checkpointpath', type=str, default=None,
                        help='Path for initialisation checkpointcheckpoint to pick up immediately from training loop. Path should be provided from the current folder..')

    args6 = parser6.parse_known_args()

    return args6[0].checkpointpath

def overdamped_argparser():
    #general configs for the training process
    # Instantiate the parser
    parser = CustomArgParse()

    parser.add_argument('-bridgetype','--bridgetype', type=str, default="half",
                        help='whether to use the phase space or marginal distributions in the terminal cost')
    
    parser.add_argument('-epochs','--epochs', type=int, default=1,
                        help='Total number of training epochs; relates to number of parameter updates for the drift (du) network')
    
    parser.add_argument('-device','--device', default="cpu",
                        help='Device to use. Currently only CPU supported')
    
    parser.add_argument('-logstep','--logstep', type=int, default=1,
                        help='Frequency to save checkpoints and compute loss metrics')
    
    parser.add_argument('-boundary','--boundary', type=str, default="UnderdampedLandauer",
                        help='which set of boundary conditions to use. See Sde.Boundary for options')
    
    parser.add_argument('-terminalcost','--terminalcost', type=str, default="GirsanovNet",
                        help='How the terminal cost should be evaluated. See Sde.terminalcost for options')
    
    #TODO... move into model module?
    parser.add_argument('-pretrain','--pretrain', type=str, default="",
                        help='Path to CSV containing a set of values to fit to the neural network before training. See model/train for parameters of the training process.')
             
    args = parser.parse_known_args(parser.argfile)

    return vars(args[0])


def integrator_params():

    # Instantiate the parser
    integrator_params = CustomArgParse()
    #subparser = parser.add_subparsers()

    integrator_params.add_argument('-batchsamples','--batchsamples', type=int, 
                        default=128,
                        help='Number of sample trajectories to take of the diffusion')

    integrator_params.add_argument('-tsteps','--tsteps', type=int, default=20,
                        help='number of timesteps in the integration')

    integrator_params.add_argument('-testsamples','--testsamples', type=int, default=2048,
                        help='number of samples for computing ref metrics')

    args1 = integrator_params.parse_known_args(integrator_params.argfile)

    return vars(args1[0])


def boundary_params():

    # Instantiate the parser
    boundary_params = CustomArgParse()
    #subparser = parser.add_subparsers()

    boundary_params.add_argument('-Tf','--Tf', type=float, default=5,
                        help='final time')

    boundary_params.add_argument('-epsilon','--epsilon', type=float, default=0.2,
                        help='constant')

    boundary_params.add_argument('-weight','--weight', type=float, default=1,
                        help='M, baldovin constant')

    boundary_params.add_argument('-g','--g', type=float, default=0,
                        help='constant')

    args2 = boundary_params.parse_known_args(boundary_params.argfile)

    return vars(args2[0])


def gaussian_params():
    #initial conditions for a gaussian half bridge
    # Instantiate the parser
    params1 = CustomArgParse()
    #subparser = parser.add_subparsers()

    params1.add_argument('-muinitial','--muinitial', type=float, default=0,
                        help='Initial value of the mean when using Gaussian boundary conditions')

    params1.add_argument('-mufinal','--mufinal', type=float, default=0,
                        help='Final value of the shift in the assigned final value function')

    params1.add_argument('-sigmainitial','--sigmainitial', type=float, default=1,
                        help='Initial value of the standard deviation when using Gaussian boundary conditions')

    params1.add_argument('-sigmafinal','--sigmafinal', type=float, default=0.25,
                        help='Final value of the scale in the assigned final value function')

    args5 = params1.parse_known_args(params1.argfile)

    return vars(args5[0])

def du_network_params():
    #general configs for the training process
    # Instantiate the parser
    parser3 = CustomArgParse()
    
    #parser3.add_argument('-checkpointpath','--checkpointpath', type=str,
    #                    default=None,
    #                    help='path to a checkpoint for warm-start training. Path should be provided from the current folder.')
         
    parser3.add_argument('-duhiddenlayerneurons','--duhiddenlayerneurons', type=int, default=16,
                        help='Number of hidden dimensions in each layer of the Neural network model')

    args3 = parser3.parse_known_args(parser3.argfile)

    return vars(args3[0])

def rho_network_params():
    #general configs for the training process
    # Instantiate the parser
    parser3 = CustomArgParse()
    
    #parser3.add_argument('-checkpointpath','--checkpointpath', type=str,
    #                    default=None,
    #                    help='path to a checkpoint for warm-start training. Path should be provided from the current folder.')
         
    parser3.add_argument('-rhohiddenlayerneurons','--rhohiddenlayerneurons', type=int, default=16,
                        help='Number of hidden dimensions in each layer of the Neural network model')

    args3 = parser3.parse_known_args(parser3.argfile)

    return vars(args3[0])

def girsanov_plot_params():

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

    # Instantiate the parser
    parser5 = argparse.ArgumentParser()
    requiredNamed = parser5.add_argument_group('required named arguments')
    requiredNamed.add_argument('-epoch','--epoch', type=int,
                                help='At which epoch checkpoint to plot. A checkpoint file ending in _epoch must be available, otherwise default initialisation is used.')
    requiredNamed.add_argument('-mcsamples','--mcsamples', type=int,default=0,
                                help='How many realisations for the mc average for Girsanov contour plots')
    args5 = parser5.parse_known_args()

    return vars(args5[0])


if __name__=="__main__":
    

    params_test = overdamped_argparser()

    print(params_test)
