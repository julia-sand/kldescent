import argparse


def overdamped_argparser():

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    parser.add_argument('-epochs','--epochs', type=int, default=70,
                        help='A required integer positional argument')
    
    parser.add_argument('-device','--device', default="cpu",
                        help='A required integer positional argument')
    
    parser.add_argument('-logstep','--logstep', type=float, default=10,
                        help='A required integer positional argument')
    
    parser.add_argument('-checkpointpath','--checkpointpath',
                        default="/results/20250424-154611/checkpoints/net_checkpoint_0.pt",    # default=None,
                        help='A required integer positional argument')

    parser.add_argument('-pre_train','--pre_train', default=False,
                        help='A required integer positional argument')
                       
    parser.add_argument('-batch_samples','--batch_samples', default=2000,
                        help='A required integer positional argument')

    parser.add_argument('-epsilon','--epsilon', type=float, default=1,
                        help='A required integer positional argument')

    parser.add_argument('-Tf','--Tf', type=float, default=1,
                        help='A required integer positional argument')

    parser.add_argument('-t_steps','--t_steps', type=float, default=20,
                        help='A required integer positional argument')

    args = parser.parse_args()

    #add validations?
    #if args.pos_arg > 10:
    #    parser.error("pos_arg cannot be larger than 10")

    return args
