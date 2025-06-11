import argparse

#TODO separate the parser into smaller so that we can parse straight into other files
def overdamped_argparser():
    #general configs for the training process
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    #subparser = parser.add_subparsers()

    parser.add_argument('-epochs','--epochs', type=int, default=1,
                        help='A required integer positional argument')
    
    parser.add_argument('-device','--device', default="cpu",
                        help='A required integer positional argument')
    
    parser.add_argument('-logstep','--logstep', type=int, default=1,
                        help='A required integer positional argument')
    
    parser.add_argument('-boundary','--boundary', type=str, default="UnderdampedHalfGaussian",
                        help='which set of boundary conditions to use. See Sde.Boundary for more details')
    
    #TODO... move into model module
    parser.add_argument('-pretrain','--pretrain', default=False,
                        help='whether or not to pretrain the network. Requires a csv containing a reference solution to be uploaded into the ref-solution module. See model/train for parameters of the training process.')
             
    args = parser.parse_known_args()

    #add validations?
    #if args.pos_arg > 10:
    #    parser.error("pos_arg cannot be larger than 10")

    return args[0]


def integrator_params():

    # Instantiate the parser
    integrator_params = argparse.ArgumentParser()
    #subparser = parser.add_subparsers()

    integrator_params.add_argument('-batchsamples','--batchsamples', type=int, 
                        default=10,
                        help='Number of sample trajectories to take of the diffusion')

    integrator_params.add_argument('-tsteps','--tsteps', type=int, default=10,
                        help='number of timesteps in the integration')
    args1 = integrator_params.parse_known_args()

    return args1[0]


def boundary_params():

    # Instantiate the parser
    boundary_params = argparse.ArgumentParser()
    #subparser = parser.add_subparsers()

    boundary_params.add_argument('-Tf','--Tf', type=float, default=5,
                        help='final time')

    boundary_params.add_argument('-beta','--beta', type=float, default=2,
                        help='constant')

    boundary_params.add_argument('-mass','--mass', type=float, default=1,
                        help='constant')

    boundary_params.add_argument('-tau','--tau', type=float, default=1,
                        help='constant')

    boundary_params.add_argument('-g','--g', type=float, default=0,
                        help='constant')

    boundary_params.add_argument('-ellsq','--ellsq', type=float, default=12.5,
                        help='typical length scale squared')
    
    args2 = boundary_params.parse_known_args()

    return args2[0]


def gaussian_params():
    #initial conditions for a gaussian half bridge
    # Instantiate the parser
    params1 = argparse.ArgumentParser()
    #subparser = parser.add_subparsers()

    params1.add_argument('-muinitial','--muinitial', type=float, default=0,
                        help='Initial value of the mean when using Gaussian boundary conditions')

    params1.add_argument('-sigmainitial','--sigmainitial', type=float, default=1,
                        help='Initial value of the standard deviation when using Gaussian boundary conditions')

    params1.add_argument('-sigmafinal','--sigmafinal', type=float, default=2,
                        help='Final value of the scale in the assigned final value function')

    params1.add_argument('-mufinal','--mufinal', type=float, default=1,
                        help='Final value of the shift in the assigned final value function')

    args5 = params1.parse_known_args()

    return args5[0]


#TODO separate the parser into smaller so that we can parse straight into other files
def network_params():
    #general configs for the training process
    # Instantiate the parser
    parser3 = argparse.ArgumentParser()
    
    parser3.add_argument('-checkpointpath','--checkpointpath', type=str,
                        default=None,
                        help='path to a checkpoint for warm-start training. Path should be provided from the current folder.')
         
    parser3.add_argument('-hiddenlayerneurons','--hiddenlayerneurons', type=int, default=8,
                        help='Number of hidden dimensions in each layer of the Neural network model')

    args3 = parser3.parse_known_args()

    #add validations?
    #if args.pos_arg > 10:
    #    parser.error("pos_arg cannot be larger than 10")

    return args3[0]


def girsanov_plot_params():

    # Instantiate the parser
    parser4 = argparse.ArgumentParser()
    
    parser4.add_argument('-xsamples','--xsamples', type=int,
                        default=80,
                        help='how many coordinates per axis. Total number of coordinates is xsamples**2')

    parser4.add_argument('-mcsamples','--mcsamples', type=int, default=10000,
                        help='Number of monte carlo trajectories to use per coordinate.')
                      
    args4 = parser4.parse_known_args()

    return args4[0]


def checkpoint_plot_params():

    # Instantiate the parser
    parser5 = argparse.ArgumentParser()
    
    parser5.add_argument('-epoch','--epoch', type=int,
                        default=2000,
                        help='At which epoch checkpoint to plot. The checkpoint file must be available.')

    parser5.add_argument('-boundary','--boundary', type=str, default="UnderdampedHalfGaussian",
                        help='Boundary conditions used in the plot')
                      

    parser5.add_argument('-folder','--folder', type=str, default=None,
                        help='Folder where to run the plots.')
            
    args5 = parser5.parse_known_args()

    return args5[0]

if __name__=="__main__":
    params_test = integrator_params()
    print(params_test.batchsamples)