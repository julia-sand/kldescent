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

class TerminalCost():
    """
    Top-level Class for methods for estimating the final value function 

    Terminal cost is the ratio of the final estimated state distribution divided by the log of the 
    reference state distribution
    g(q,p) = log(^P(q,p) / Pref)
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
    
    def gfun(self,x):
        
        pdf = self.get_logpdf(x)
        return pdf + self.boundary.log_pfinal(x[...,[0]],x[...,[1]])


class GirsanovNet(TerminalCost):
    """
    Estimate the final cost with a neural network trained on the final distribution
    Note this is only available for full final distribution
    """

    def __init__(self,boundaryconditions,logger,*_):
        """
        Initialise a neural network for learning the final distribution via Girsanov theorem 
        
        args:
            boundaryconditions: class
        """
        
        from ..model.nn import RhoNeuralNetwork
        from ..model.train import TrainRho

        TerminalCost.__init__(self,boundaryconditions,logger)
        
        rhonet = RhoNeuralNetwork() 
        
        self.rhonet = DataParallel(rhonet) 
        self.logger.log_dict(self.rhonet.module.params)
        
        #initialise a trainer class for rho
        self.rhotrainer = TrainRho(self.rhonet)

    def update_cost_params(self,integrator,epoch):
        #train the rho neural network
        self.rhotrainer.train_loop(integrator)

        #save the checkpoint
        self.logger.save_checkpoint(self.rhonet,epoch)

    def get_logpdf(self,x):
        return torch.nan_to_num(self.rhonet(x,condition="log"))

    #def gfun(self,x):
    #    pdf = 
    #    return torch.nan_to_num(pdf) + self.boundary.log_pfinal(x[...,[0]],x[...,[1]])
        #(self.boundary.Ufinal(x[...,[0]])+((x[...,[1]]**2)/2))

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

        #initialise variables
        self.pos_mean0 = 0 
        self.pos_var0 = 1
        self.mom_mean0 = 0 
        self.mom_var0 = 1
        self.cc0 = 0 

    def update_cost_params(self,integrator,*_):
        #compute x vals
        x = integrator.update_ref_statistics()

        #set new mean and variance and covariance
        self.pos_mean0 = torch.nanmean(x[...,0]).item() 
        self.pos_var0 = torch.var(x[...,0]).item()
        self.mom_mean0 = torch.nanmean(x[...,1]).item() 
        self.mom_var0 = torch.var(x[...,1]).item()
        self.cc0 = (torch.sum((x[...,1]-self.mom_mean0)*(x[...,0]-self.pos_mean0))/(len(x[...,0]))).item()

    def multivariate_U(self,x):
        #exponent of a multivariate Gaussian distribution
        #pdb.set_trace()
        rho_var = self.cc0/(self.mom_var0*self.pos_var0)
        return -(1/(2*(1-rho_var**2)))*(((x[...,[0]]-self.pos_mean0)**2)/self.pos_var0 - 2*(self.cc0*(x[...,[0]]-self.pos_mean0)*(x[...,[1]]-self.mom_mean0)) + ((x[...,[1]]-self.mom_mean0)**2)/self.mom_mean0)

    def get_logpdf(self,x):

        return self.multivariate_U(x)

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
                                            bins=500,
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
        x = datapoint[0].numpy()
        y = datapoint[1].numpy()

        #compare datapoint with bin edges
        indx = np.nonzero(self.xedges<x)[0][-1]# and x < self.xedges))
        indy = np.nonzero(self.yedges<y)[0][-1]# and y < self.yedges))

        #step 2: return the correct histogram value
        return self.H[indy-1,indx-1]

    def get_half_logpdf(self,x):
        pdf = torch.tensor([self.evaluate_density(x[i,:]) for i in range(len(x[:,0]))],dtype=torch.float32,requires_grad=False)
        return torch.nan_to_num(torch.log(pdf+1e-10))

    def get_logpdf(self,x):
        pdf = torch.tensor([self.evaluate_density(x[i,:]) for i in range(len(x[:,0]))],dtype=torch.float32,requires_grad=False)
        return torch.nan_to_num(torch.log(pdf+1e-10))

    #def gfun(self,x):
        
    #    pdf = torch.tensor([self.evaluate_density(x[i,:]) for i in range(len(x[:,0]))],dtype=torch.float32,requires_grad=False)
    #    return torch.nan_to_num(torch.log(pdf+1e-10)) + self.boundary.log_pfinal(x[...,[0]],x[...,[1]])


class NormFlow(TerminalCost):

    def __init__(self,boundaryconditions,logger):
        from ..model.normflow import RhoNormFlow,Train_normflow

        TerminalCost.__init__(self,boundaryconditions,logger)
        #initialise normalising flow
        self.rho = RhoNormFlow()
        self.trainer_normflow = Train_normflow(self.rho.model,self.boundary)

    def update_cost_params(self,integrator,*_):

        #integrate and get final samples
        x = integrator.update_ref_statistics()
        self.trainer_normflow.train_loop(integrator,x)


    def get_half_logpdf(self,x):
        pdf = self.rho.model.log_prob(x[...,0])
        return torch.nan_to_num(pdf) 

    def get_logpdf(self,x):
        pdf = self.rho.model.log_prob(x)
        return torch.nan_to_num(pdf) 