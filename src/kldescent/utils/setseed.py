import numpy as np
import torch
import random

from kldescent.utils import parser


def set_seed(s=None):
    """
    Set the seed for random sampling in numpy and torch
    If no seed is given, use command line argument
    """

    if s is None:
        s = parser.seed()    
        
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    return 

if __name__ is "__main__":
    set_seed()