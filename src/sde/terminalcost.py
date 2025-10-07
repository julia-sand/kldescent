"""
This file contains methods for estimating the final value function 
in the half bridge. 

Methods 
    normflow: Use a neural network to learn the final distribution
    MLE: When we know the final distribution is Gaussian, we can use a parametric method of distirbution estimation, namely estimate first and second moments by MLE. 
    GirsanovNet: Use a neural network to learn the final distribution 
    histogram: Use a histogram to estimate the final value function
"""

import numpy as np 
import torch
from torch.nn import DataParallel
#from torch import nn

from ..model.nn import RhoNeuralNetwork
from ..model.train import TrainRho 
from .fpk import FokkerPlanck

class TerminalCost():
    """
    Top-level Class for methods for estimating the final value function 
    """

    def __init__(self,boundaryconditions,logger):
        """
        Initialise how the terminal cost should be evaluated.    
        args:
            boundaryconditions: 
                associated boundary conditions
            logger: class
                read and write saved run data (eg checkpoints)
        """

        self.boundary = boundaryconditions
        self.logger = logger

    def marginal_halfbridge(self,):
        """
        Terminal cost is the ratio of the final estimated configuration distribution divided by the log of the 
        reference configuration distribution
        g(q,p) = log(^P_marg(q,p) / Pref_marg)
        """

        pass

    def quarterbridge(self,):
        pass

    def full_halfbridge(self,):
        """
        Terminal cost is the ratio of the final estimated state distribution divided by the log of the 
        reference state distribution
        g(q,p) = log(^P(q,p) / Pref)
        """
        
        pass

class GirsanovNet(TerminalCost):
    """
    Estimate the final cost with a neural network trained on the final distribution
    """

    def __init__(self,boundaryconditions,logger,*_):
        """
        Initialise a neural network for learning the final distribution via Girsanov theorem 
        
        args:
            boundaryconditions: class
        """

        TerminalCost.__init__(self,boundaryconditions,logger)
        
        rhonet = RhoNeuralNetwork() 
        
        self.rhonet = DataParallel(rhonet) 
        self.logger.log_dict(terminal_cost.rhonet.module.params)

    def update_cost_params(self,integrator,epoch):
        #initialise the fokkerplanck integration
        fpk_integrator = FokkerPlanck(integrator.boundary,integrator.net)
        
        #initialise a trainer class for rho
        rhotrainer = TrainRho(self.rhonet,fpk_integrator)

        #train the rho neural network
        rhotrainer.train_loop()

        #save the checkpoint
        self.logger.save_checkpoint(self.rhonet,epoch)

    def gfun(self,x):
        pdf = self.rhonet(x,condition="log")
        return torch.nan_to_num(pdf) + (self.boundary.Ufinal(x[...,[0]])+((x[...,[1]]**2)/2))

class MLE(TerminalCost):
    """
    Finds the final value function using a parametric estimate of the distribution as a gaussian and MLE 
    estimates for the mean and variance
    """

    def __init__(self,boundaryconditions,logger,*_):
        """
        Initialise placeholder parameters for estimating the final distirbution using a parametric method

        args:
            boundaryconditions: class
        """

        TerminalCost.__init__(self,boundaryconditions,logger)
        #self.integrator = integrator
        self.mean0 = 0 
        self.var0 = 1

    def update_cost_params(self,integrator,*_):
        #compute x vals
        x = integrator.update_ref_statistics()

        #set new mean and variance
        self.mean0 =torch.nanmean(xinit[...,0]).item()
        self.var0 = torch.var(xinit[...,0]).item()
    
    def Utemp(self,q,mean,var):

        return (q-mean)**2/(var)

    def gfun(self,x):

        return (self.boundary.params["weight"]/2)*(-self.Utemp(x[...,[0]],self.mean0,self.var0)+self.Utemp(x[...,[0]],self.boundary.gauss_boundary["mufinal"], self.boundary.gauss_boundary["sigmafinal"]**2))

class Histogram(TerminalCost):
    """
    Find the final probabilities from an estimated histogram
    """

    def __init__(self,boundaryconditions,logger):

        TerminalCost.__init__(self,boundaryconditions,logger)
        
        #initialise blank histogram values
        self.H = None
        self.yedges = None
        self.xedges = None 

    def update_cost_params(self,integrator,*_):
        x = integrator.update_ref_statistics().numpy()
        H, yedges, xedges = np.histogram2d(x[...,1], x[...,0], 
                                            bins=40,
                                            range=[[-5, 5], [-5, 5]],
                                            density=True)
        self.H = H
        self.yedges = yedges
        self.xedges = xedges
        return 

    def evaluate_density(self, datapoint):
        """
        args:
            datapoint: data point or array of points in the shape (2,)
        returns: 
            p1: estimate of probability density from histogram
        """

        #step 1: check which bin for x and y the point is in
        #get x,y coordinate
        x = datapoint[0]
        y = datapoint[1]

        #compare datapoint with bin edges
        indx = np.nonzero(self.xedges<x)[0][-1]# and x < self.xedges))
        indy = np.nonzero(self.yedges<y)[0][-1]# and y < self.yedges))
        #print(f"{x}",f"{y}")

        #print("pdf=",self.H[indx,indy])
        #step 2: return the correct histogram value
        return self.H[indy-1,indx-1]

    def gfun(self,x):
        
        pdf = torch.array([self.evaluate_density(x[i,:]) for i in range(len(x[:,0]))])
        return np.nan_to_num(np.log(pdf+1e-10)) + (self.Ufinal(x[...,[0]])+((x[...,[1]]**2)/2))

class NormFlow(TerminalCost):

    def __init__(self,boundaryconditions,logger):

        TerminalCost.__init__(self,boundaryconditions,logger)

    def update_cost_params(self,integrator):

        #TODO train the normalising flow here? 

        pass

    def gfun(self,x):
        #pdf = self.rhonet(x,condition="log")
        #return torch.nan_to_num(pdf) + (self.boundary.Ufinal(x[...,[0]])+((x[...,[1]]**2)/2))
        pass