"""
This file contains methods for estimating the final value function 
in the half bridge. 

Methods 
    MLE: When we know the final distribution is Gaussian, we can use a parametric method of distirbution estimation, namely estimate first and second moments by MLE. 
    histogram: Use a histogram to estimate the final distribution of the positions
"""

import numpy as np 
import torch

from ..viz.distribution_plots import *
from ..utils.logger import save_params

class TerminalCost():
    """
    Top-level Class for methods for estimating the final value function 
    
    Terminal cost is the ratio of the final estimated configuration distribution divided by the log of the 
    reference configuration distribution
    g(q,p) = log(^P_marg(q,p) / Pref_marg)
    """

    def __init__(self,boundaryconditions,
                        fileloc):
        """
        Initialise how the terminal cost should be evaluated.    
        args:
            boundaryconditions: 
                associated boundary conditions
            fileloc: str
                save parameters
        """

        self.boundary = boundaryconditions
        self.M = self.boundary.params["weight"]
        self.loc = fileloc
    
    def gfun(self,x):
        
        pdf = self.get_half_logpdf(x)
        return self.M*(pdf + self.boundary.Ufinal(x[...,[0]]))
        
class GirsanovNet(TerminalCost):
    """
    Estimate the final cost with a neural network trained on the final distribution
    Note this is only available for full final distribution
    This class is used only to catch the error explicitly
    """

    def __init__(self,*_):
        
        raise NotImplementedError("Method not available for marginal half-bridge")
        
class MLE(TerminalCost):
    """
    Finds the final value function using a parametric estimate of the distribution as a gaussian and MLE 
    estimates for the mean and variance
    """

    def __init__(self,boundaryconditions,fileloc,*_):
        """
        Initialise placeholder parameters for estimating the final distirbution using a parametric method

        args:
            boundaryconditions : class
            fileloc : folder
        """

        TerminalCost.__init__(self,boundaryconditions,fileloc)

        #initialise variables
        self.pos_mean0 = 0 
        self.pos_var0 = 1

    def update_cost_params(self,integrator,*_):
        """compute cumulants for the current final distribution"""

        #compute x vals
        x = integrator.update_ref_statistics()

        #set new mean and variance and covariance
        self.pos_mean0 = torch.nanmean(x[...,0]).item() 
        self.pos_var0 = torch.var(x[...,0]).item()
               
    def get_half_logpdf(self,x):

        return -((x[...,[0]]-self.pos_mean0)**2)/(2*self.pos_var0)

class Histogram(TerminalCost):
    """Find the final probabilities from an estimated histogram"""

    def __init__(self,boundaryconditions,fileloc):

        TerminalCost.__init__(self,boundaryconditions,fileloc)
        
        #initialise blank histogram values
        self.H = None
        self.xedges = None 

        self.nbins = 100
        self.xrange = [-5,5]

    def update_cost_params(self,integrator,epoch,*_):
        """compute a histogram for the final distribution"""

        x = integrator.update_ref_statistics().cpu().numpy()
        H, xedges = np.histogram(x[...,0], 
                                            bins=self.nbins,
                                            range=self.xrange,
                                            density=True)
        self.H = H
        self.xedges = xedges

        #plot learned distribution vs histogram
        #plot_loc = self.logger.newpath 
        #plot_histogram(plot_loc,self.H,self.xedges,epoch)

        return 

    def evaluate_density(self, x):
        """
        args:
            x: position sample point or array of points in the shape (1,)
        returns: 
            p1: estimate of probability density from histogram
        """

        #step 1: check which bin for x and y the point is in
        #compare datapoint with bin edges
        indx = np.nonzero(self.xedges<x)[0][-1]# and x < self.xedges))

        #step 2: return the correct histogram value
        return self.H[indx-1]

    def get_half_logpdf(self,x):
        pdf = torch.tensor([self.evaluate_density(x[i,0].numpy()) for i in range(len(x[:,0]))],dtype=torch.float32,requires_grad=False)
        return torch.nan_to_num(torch.log(pdf+1e-10))

class NormFlow(TerminalCost):

    def __init__(self,boundaryconditions,fileloc):
        from ..model.normflow import RhoNormFlow

        TerminalCost.__init__(self,boundaryconditions,fileloc)
        
        #initialise normalising flow
        self.rho = RhoNormFlow(boundaryconditions=self.boundary,bridgetype="marginal")

    def update_cost_params(self,integrator,*_):
        """train a normalising flow to approximate the final distribution"""

        #get initial samples
        x0 = torch.tensor(self.boundary.generate_samples(integrator.sde_params["testsamples"] ,"initial"),requires_grad=False,dtype=torch.float32)
        
        #integrate and get final samples
        x = torch.tensor(self.boundary.generate_samples(8192,"final"),dtype=torch.float32)
        #integrator.sde_params["testsamples"] ,"final"),dtype=torch.float32)

        #put model in train mode
        self.rho.model.train()

        #train normalising flow
        self.rho.train_loop(x[...,0])
        
        #put model in evaluation mode
        self.rho.model.eval()
        
        #plot learned distribution vs histogram
        #plot_loc = self.logger.newpath 
        #make_histogram(plot_loc,x,self.rho.model,self.boundary)

    def get_half_logpdf(self,x):
        pdf = self.rho.model.log_prob(x[...,[0]])
        return torch.nan_to_num(pdf) 

    @torch.no_grad
    def make_plot(self,x,location):
        #TODO fix this
        import matplotlib.pyplot as plt

        #plot_loc = self.logger.newpath 
        
        #make a plot of final points vs estimated distribution
        pdf_temp = torch.exp(self.get_half_logpdf(x))
        pdf_temp[torch.isnan(pdf_temp)] = 0
        plt.scatter(x[...,0],pdf_temp.data.numpy())
        plt.hist(x[...,0],density=True,bins=100,alpha=0.5,color="orange")
        plt.savefig(plot_loc+"/distribution.png")
        plt.close()
        