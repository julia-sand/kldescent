"""Tests the run time of the statistics calculations"""

import numpy as np
import torch

from src.model import *
from src.sde import *
import src.sde.boundary as boundaries
from src.utils import *
#from src.utils.parser import integrator_params

#@profile
def run(testing_samples):
    
    #initialise the model
    #TODO: add gpu option
    #TODO: add config file to read these params
    #dynamically create boundary class bsaed on input
    boundaryconditions = boundaries.UnderdampedHalfGaussian()#getattr(boundaries,"UnderdampedHalfGaussian")()
    net = nn.NeuralNetwork(boundaryconditions).to("cpu")
 
    #intialise sde integration 
    #TODO add these arguments so they get read directly from the command line already in the sde file
    integrator = langevin.SDE(boundaryconditions)

    integrator.update_ref_statistics(net,testing_samples)

if __name__=="__main__":
    run(testing_samples)
