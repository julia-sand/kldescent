"""
trains a simple neural network to interpolate the estimated
perturbative drift from the csv
"""

import torch

from ...model.nn import MLP
from ...model.pretrain import run_pretrain
from ...sde.boundary import *
from ..reference import *

def perturbative_du_interp():

    #set up boundary conditions
    boundary = UnderdampedLandauer()

    #get the reference solution 
    reference_sol = RefSolution(boundary)
       
    #initialise a neural net
    mlp_net = MLP()

    #train on the csv
    run_pretrain(mlp_net,reference_sol.perturb_underdamped_loc)

    return mlp_net



if __name__=="__main__":
    mlp_net = perturbative_du_interp()