import os

import torch

from kldescent.model.train import Pretrain

def run_pretrain(net,benchmark=None,
                    path_to_pretrain=None):
    """
    Trains a neural network to approximate a numerical function given by values in a csv

    args:
        net:
            neural network to be trained
        benchmark : class 
            sets the path to pretain as the location of the reference solution csv 
        path_to_pretrain : csv
            path from current working directory to a CSV containing data to be used in the pretraining. The data should have a csv format 
            where the first two entries are the input and the last column should contian the trained values. See PerturbativeBenchmarkSolution folder for some examples.  
    returns:
        none
    """

    print("Starting pre-training")
    
    pretrainer = Pretrain(benchmark=benchmark,path_to_pretrain=path_to_pretrain)
    pretrainer.pretraining(net)
    
    print("Pretraining completed successfully.")

if __name__=="__main__":
    run_pretrain()