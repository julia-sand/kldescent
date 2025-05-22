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
    
    parser.add_argument('-logstep','--logstep', type=int, default=100,
                        help='A required integer positional argument')
    
    parser.add_argument('-checkpointpath','--checkpointpath', type=str,
                        default=None,
                        #default="zeros",
                        #default="/results/20250429-195812/checkpoints/net_checkpoint_0.pt",    # default=None,
                        help='path to a checkpoint for warm-start training. Path should be provided from the current folder.')

    parser.add_argument('-pretrain','--pretrain', default=False,
                        help='whether or not to pretrain the network. Requires a csv containing a reference solution to be uploaded into the ref-solution module. See model/train for parameters of the training process.')
                      
    parser.add_argument('-hiddenlayerneurons','--hiddenlayerneurons', type=int, default=10,
                        help='Number of hidden dimensions in each layer of the Neural network model')

    parser.add_argument('-boundary','--boundary', type=str, default="UnderdampedLandauer",
                        help='which set of boundary conditions to use. See Sde.Boundary for more details')
    
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

    boundary_params.add_argument('-Tf','--Tf', type=float, default=1,
                        help='final time')

    boundary_params.add_argument('-beta','--beta', type=float, default=2,
                        help='constant')

    boundary_params.add_argument('-mass','--mass', type=float, default=1,
                        help='constant')

    boundary_params.add_argument('-tau','--tau', type=float, default=1,
                        help='constant')

    boundary_params.add_argument('-g','--g', type=float, default=0,
                        help='constant')

    args2 = boundary_params.parse_known_args()

    return args2[0]

if __name__=="__main__":
    params_test = integrator_params()
    print(params_test.batchsamples)