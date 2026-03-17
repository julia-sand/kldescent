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

from .kl import kl_div
from ..viz.distribution_plots import *
from ..model.nn import save_nn_checkpoint
from ..utils.logger import save_params
from .girsanovpdf import compute_pdf

class TerminalCost():
    """
    Top-level Class for methods for estimating the final value function 

    Terminal cost is the ratio of the final estimated distribution in phase space divided by the log of the 
    reference phase space distribution
    g(q,p) = log(^P(q,p) / Pref)
    """

    def __init__(self,boundaryconditions,fileloc,*_):
        """
        Initialise how the terminal cost should be evaluated.    
        args:
            boundaryconditions: 
                associated boundary conditions
            filelocation: str
                filepath to results folder, for saved checkpoints
            filename    :   str
                file name for saving params  
        """

        self.boundary = boundaryconditions
        self.loc = fileloc
        #self.datafile = filename
        self.M = self.boundary.params["weight"]

    def gfun(self,x,*_):
        
        pdf = torch.nan_to_num(self.get_logpdf(x),posinf=0,neginf=0,nan=0)
        return self.M*(pdf + self.boundary.log_pfinal(x[...,[0]],x[...,[1]]))

class GirsanovMC(TerminalCost):
    """
    Estimate the final distribution by a Monte Carlo method
    """
    def __init__(self,boundaryconditions,fileloc,filename=None,*_):
        """
        args:
            boundaryconditions: class
            filelocation : str
        """
        
        TerminalCost.__init__(self,boundaryconditions,fileloc,*_)
        
    def update_cost_params(self,integrator,epoch):
        """compute the loss first for this method"""
        pass

    def gfun(self,x,net,mc_samples,*_):
        
        pdf = torch.nan_to_num(self.get_logpdf(x,net,mc_samples),posinf=0,neginf=0,nan=0)
        return self.M*(pdf + self.boundary.log_pfinal(x[...,[0]],x[...,[1]]))

    def get_logpdf(self,x,net,mc_samples):
        
        pdf = torch.zeros_like(x[...,[0]])    
        for i in range(pdf.shape[0]):
            pdf[i] = torch.mean(compute_pdf(x[i,None],net,
                                    self.boundary.sigma0,
                                    self.boundary.params["epsilon"],
                                    self.boundary.p_initial,
                                    self.boundary.params["Tf"],
                                    mc_samples))
        #print(pdf)
        return torch.log(pdf)


class GirsanovNet(TerminalCost):
    """
    Estimate the final cost with a neural network trained on the final distribution
    Note this is only available for full final distribution
    """

    def __init__(self,boundaryconditions,fileloc,filename):
        """
        Initialise a neural network for learning the final distribution via Girsanov theorem 
        
        args:
            boundaryconditions: class
            filelocation : str
        """
        
        from ..model.nn import RhoNeuralNetwork
        from ..model.train import TrainRho

        TerminalCost.__init__(self,boundaryconditions,fileloc)
        
        self.rhonet = RhoNeuralNetwork() 
        
        save_params(self.rhonet.params,filename)
        
        #initialise a trainer class for rho
        self.rhotrainer = TrainRho(self.rhonet)

    def update_cost_params(self,integrator,epoch):
        """train girsanov net to find the final distribution"""

        if epoch<1:
            pass
        else: #update number of epochs
            self.rhotrainer.rho_epochs = 1024 #less epochs for training later 

        #train the rho neural network
        self.rhotrainer.train_loop(integrator)

        #save the checkpoint
        save_nn_checkpoint(self.rhonet,loc=self.loc+f"/checkpoints/{self.rhonet.net_name}_checkpoint_{epoch+1}.pt")

    def get_logpdf(self,x,*_):
        return torch.nan_to_num(self.rhonet(x,condition="log"))

class MLE(TerminalCost):
    """
    Finds the final value function using a parametric estimate of the distribution as a gaussian and MLE 
    estimates for the mean and variance
    """

    def __init__(self,boundaryconditions,fileloc,filename=None,*_):
        """
        Initialise placeholder parameters for estimating the final distirbution using a parametric method

        args:
            boundaryconditions: class
        """

        TerminalCost.__init__(self,boundaryconditions,fileloc,*_)

        #initialise variables
        self.pos_mean0 = 0 
        self.pos_var0 = 1
        self.mom_mean0 = 0 
        self.mom_var0 = 1
        self.cc0 = 0 

    def update_cost_params(self,integrator,*_):
        """update the cumulants for approximating the final distribution with a gaussian"""

        #compute x vals
        x = integrator.update_ref_statistics()

        #set new mean and variance and covariance
        self.pos_mean0 = torch.nanmean(x[...,0]).item() 
        self.mom_mean0 = torch.nanmean(x[...,1]).item() 

        #self.Sigma_matrix = torch.cov(x)

        self.pos_var0 = torch.var(x[...,0]).item()
        self.mom_var0 = torch.var(x[...,1]).item()

        self.cc0 = (torch.sum((x[...,1]-self.mom_mean0)*(x[...,0]-self.pos_mean0))/(len(x[...,0]))).item()

    def multivariate_U(self,x):
        #exponent of a multivariate Gaussian distribution
        rho_var = self.cc0/np.sqrt(self.mom_var0*self.pos_var0)
        return -(1/(2*(1-rho_var**2)))*(((x[...,[0]]-self.pos_mean0)**2)/self.pos_var0 - 2*((x[...,[0]]-self.pos_mean0)*(x[...,[1]]-self.mom_mean0))*(self.cc0/(self.mom_var0*self.pos_var0)) + ((x[...,[1]]-self.mom_mean0)**2)/self.mom_var0)

    def get_logpdf(self,x,*_):

        return self.multivariate_U(x)

class Histogram(TerminalCost):
    """
    Find the final probabilities from an estimated histogram
    """

    def __init__(self,boundaryconditions,fileloc,filename=None,*_):

        TerminalCost.__init__(self,boundaryconditions,fileloc,*_)
        
        #initialise blank histogram values
        self.H = None
        self.yedges = None
        self.xedges = None 

    def update_cost_params(self,integrator,*_):
        """make histogram of the final distribution"""

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

    def get_logpdf(self,x,*_):
        pdf = torch.tensor([self.evaluate_density(x[i,:]) for i in range(len(x[:,0]))],dtype=torch.float32,requires_grad=False)
        return torch.nan_to_num(torch.log(pdf+1e-10))

class NormFlow(TerminalCost):

    def __init__(self,boundaryconditions,fileloc,filename=None,*_):
        from ..model.normflow import RhoNormFlow

        TerminalCost.__init__(self,boundaryconditions,fileloc)
        #initialise normalising flow
        self.rho = RhoNormFlow(boundaryconditions=self.boundary,bridgetype="half")

    def update_cost_params(self,integrator,epoch,*_):
        """Train the normalising flow to approximate the final distribution"""

        if epoch<1:
            pass
        else: #update number of epochs
            self.rho.max_iter = 1000 #less epochs for training later 

        
        #get initial samples
        x0 = torch.tensor(self.boundary.generate_samples(integrator.sde_params["testsamples"] ,"initial"),requires_grad=False,dtype=torch.float32)
        
        #integrate and get final samples
        x = integrator.update_ref_statistics(x0_samples=x0)

        #put model in train mode
        self.rho.model.train()

        #train normalising flow
        self.rho.train_loop(x)
        
        #put model in evaluation mode
        self.rho.model.eval()

        plot_loc = self.loc 
        make_scatter_plot(plot_loc,x,self.rho.model,epoch)#,self.boundary)
        distribution_plot(plot_loc,self.rho.model,self.boundary.params["Tf"],epoch,self.boundary)

    def get_logpdf(self,x,*_):
        pdf = self.rho.model.log_prob(x)
        return torch.nan_to_num(pdf)     