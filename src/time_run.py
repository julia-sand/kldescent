"""Tests the run time of the statistics calculations"""

import os

import numpy as np
import torch

from src.model import *
from src.sde import *
import src.sde.boundary as boundaries
from src.utils import *

#@profile
def run():

    #initialise the model
    #TODO: add gpu option, decide architectures
    #TODO: add config file to read these params
    #dynamically create boundary class bsaed on input
    boundaryconditions = getattr(boundaries,"UnderdampedHalfGaussian")()
    net = nn.NeuralNetwork(boundaryconditions).to("cpu")
 
    #intialise sde integration 
    #TODO add these arguments so they get read directly from the command line already in the sde file
    integrator = langevin.SDEOverdamped(boundaryconditions)

    integrator.update_ref_statistics(net,testing_samples = 10)

if __name__=="__main__":
    run()
