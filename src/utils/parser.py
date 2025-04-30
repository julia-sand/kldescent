import argparse

#TODO separate the parser into smaller so that we can parse straight into other files
def overdamped_argparser():

    # Instantiate the parser
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    parser.add_argument('-epochs','--epochs', type=int, default=1,
                        help='A required integer positional argument')
    
    parser.add_argument('-device','--device', default="cpu",
                        help='A required integer positional argument')
    
    parser.add_argument('-logstep','--logstep', type=int, default=10,
                        help='A required integer positional argument')
    
    parser.add_argument('-checkpointpath','--checkpointpath', type=str,
                        default=None,
                        #default="zeros",
                        #default="/results/20250429-195812/checkpoints/net_checkpoint_0.pt",    # default=None,
                        help='A required integer positional argument')

    parser.add_argument('-pretrain','--pretrain', default=False,
                        help='A required integer positional argument')
                       
    parser.add_argument('-batch_samples','--batch_samples', type=int, 
                        default=600,
                        help='A required integer positional argument')

    parser.add_argument('-epsilon','--epsilon', type=float, default=1,
                        help='A required integer positional argument')

    parser.add_argument('-Tf','--Tf', type=float, default=1,
                        help='A required integer positional argument')

    parser.add_argument('-t_steps','--t_steps', type=int, default=20,
                        help='A required integer positional argument')

    args = parser.parse_args()

    #add validations?
    #if args.pos_arg > 10:
    #    parser.error("pos_arg cannot be larger than 10")

    return args
