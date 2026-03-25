"""Tests the run time of the statistics calculations"""

import numpy as np
import torch

from kldescent.model import *
from kldescent.sde import *
import kldescent.sde.boundary as boundaries
from kldescent.utils import *

def run(testing_samples):
    
    #initialise the model
    #TODO: add gpu option
    #dynamically create boundary class bsaed on input
    boundaryconditions = boundaries.UnderdampedHalfGaussian()#getattr(boundaries,"UnderdampedHalfGaussian")()
    net = nn.NeuralNetwork(boundaryconditions).to("cpu")
 
    #intialise sde integration 
    #TODO add these arguments so they get read directly from the command line already in the sde file
    integrator = langevin.SDE(boundaryconditions)

    integrator.update_ref_statistics(testing_samples)

if __name__=="__main__":
    run(testing_samples)
