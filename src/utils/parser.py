import argparse


def overdamped_argparser():

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('epochs', type=int, default=70,
                        help='A required integer positional argument')
    
    parser.add_argument('device', default="cpu",
                        help='A required integer positional argument')
    
    parser.add_argument('logstep', type=float, default=10,
                        help='A required integer positional argument')
    parser.add_argument('checkpointpath', default=None,
                        help='A required integer positional argument')

    parser.add_argument('pre_train', default=False,
                        help='A required integer positional argument')
                        
    parser.add_argument('path_to_pretrain', default="src/refsolution/value.csv",
                        help='A required integer positional argument')
                        
    parser.add_argument('plot_times', default=[0,0.5,1.0],
                        help='A required integer positional argument')

    parser.add_argument('batch_samples', default=2000,
                        help='A required integer positional argument')

    parser.add_argument('epsilon', type=float, default=1,
                        help='A required integer positional argument')

    parser.add_argument('Tf', type=float, default=1,
                        help='A required integer positional argument')


    parser.add_argument('t_steps', type=float, default=20,
                        help='A required integer positional argument')

    args = parser.parse_args()

    #add validations?
    #if args.pos_arg > 10:
    #    parser.error("pos_arg cannot be larger than 10")

    return parser 