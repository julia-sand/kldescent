"""
This file contains methods for estimating the final value function 
in the half bridge. 

Classes 
    normflow: Use a neural network to learn the final distribution
    MLE: When we know the final distribution is Gaussian, we can use a parametric method of distirbution estimation, namely estimate first and second moments by MLE. 
    GirsanovNet: Use a neural network to learn the final distribution 
    histogram: Use a histogram to estimate the final value function
"""

import numpy as np
import torch

from kldescent.viz.distribution_plots import *
from kldescent.model.nn import save_nn_checkpoint
from kldescent.utils.logger import save_params
from kldescent.sde.girsanovpdf import compute_log_pdf

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

        Parameters
        ----------
        boundaryconditions : object
            Associated boundary conditions.
        fileloc : str
            Filepath to results folder, for saved checkpoints.
        """

        self.boundary = boundaryconditions
        
        self.d = self.boundary.d

        self.loc = fileloc
        #self.datafile = filename
        self.Mtarget = self.boundary.params["weight"]
        self.ramp = 30
        self.M = 0

    def update_M(self,epoch):
        
        self.M = self.Mtarget * min(1.0, (epoch / self.ramp)) ** 2 
        

    def gfun(self,x,*_):
        
        pdf = torch.nan_to_num(self.get_logpdf(x),posinf=0,neginf=0,nan=0)
        return self.M*(pdf + self.boundary.log_pfinal(x[...,:self.d],x[...,[1]]))

class GirsanovMC(TerminalCost):
    """
    Estimate the final distribution by a Monte Carlo method
    """
    def __init__(self,boundaryconditions,fileloc,filename=None,*_):
        """
        Parameters
        ----------
        boundaryconditions : object
            Associated boundary conditions.
        fileloc : str
            Filepath to results folder.
        """

        TerminalCost.__init__(self,boundaryconditions,fileloc,*_)

    def update_cost_params(self,integrator,epoch):
        """compute the loss first for this method"""
        self.update_M(epoch)
        
    def gfun(self, x, integrator, mc_samples, *_):

        pdf = torch.nan_to_num(self.get_logpdf(x, integrator, mc_samples), posinf=0, neginf=0, nan=0)
        return self.M*(pdf + self.boundary.log_pfinal(x[...,:self.d], x[...,[1]]))

    def get_logpdf(self, x, integrator, mc_samples):

        return compute_log_pdf(x, integrator, mc_samples=mc_samples)

class GirsanovNet(TerminalCost):
    """
    Estimate the final cost with a neural network trained on the final distribution
    Note this is only available for full final distribution
    """

    def __init__(self,boundaryconditions,fileloc,filename):
        """
        Initialise a neural network for learning the final distribution via Girsanov theorem.

        Parameters
        ----------
        boundaryconditions : object
            Associated boundary conditions.
        fileloc : str
            Filepath to results folder.
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
        self.update_M(epoch)

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
        Initialise placeholder parameters for estimating the final distribution
        using a parametric method.

        Parameters
        ----------
        boundaryconditions : object
            Associated boundary conditions.
        """

        TerminalCost.__init__(self,boundaryconditions,fileloc,*_)

        #initialise variables
        self.pos_mean0 = 0 
        self.pos_var0 = 1
        self.mom_mean0 = 0 
        self.mom_var0 = 1
        self.cc0 = 0 

    def update_cost_params(self,integrator,epoch):
        """update the cumulants for approximating the final distribution with a gaussian"""
        self.update_M(epoch)

        #compute x vals
        x = integrator.update_ref_statistics()

        #set new mean and variance and covariance
        self.pos_mean0 = torch.nanmean(x[...,:self.d])
        self.mom_mean0 = torch.nanmean(x[...,[1]])

        #self.Sigma_matrix = torch.cov(x)

        self.pos_var0 = torch.var(x[...,:self.d])
        self.mom_var0 = torch.var(x[...,[1]])

        self.cc0 = (torch.sum((x[...,[1]]-self.mom_mean0)*(x[...,:self.d]-self.pos_mean0))/(len(x[...,0])))


    def multivariate_U(self,x):
        r"""Exponent of a multivariate Gaussian distirbution
        
        With $q = $ x[...,[:d]], $p = $ x[...,[d:]] and estimated statistics $\mu_q, \mu_p, \sigma_q^2, \sigma_p^2, c_0 = \widehat{\text{Cov}}(q,p)$:

        $$\rho = \frac{c_0}{\sqrt{\sigma_q^2,\sigma_p^2}}$$

        $$U(q,p) = -\frac{1}{2(1-\rho^2)}\left[\frac{(q-\mu_q)^2}{\sigma_q^2} - \frac{2\rho,(q-\mu_q)(p-\mu_p)}{\sigma_q,\sigma_p} + \frac{(p-\mu_p)^2}{\sigma_p^2}\right]$$

        This is the exponent of a bivariate Gaussian (i.e. $\log \rho_f$ up to the normalising constant).
        Note currently only implemented for d=1
        """
        
        rho_var = self.cc0/torch.sqrt(self.mom_var0*self.pos_var0)
        return -(1/(2*(1-rho_var**2)))*(((x[...,:self.d]-self.pos_mean0)**2)/self.pos_var0 - 2*((x[...,:self.d]-self.pos_mean0)*(x[...,self.d:]-self.mom_mean0))*(self.cc0/(self.mom_var0*self.pos_var0)) + ((x[...,self.d:]-self.mom_mean0)**2)/self.mom_var0)

    def get_logpdf(self,x,*_):

        return self.multivariate_U(x)

class Histogram(TerminalCost):
    """
    Find the final probabilities from an estimated histogram
    """

    def __init__(self,boundaryconditions,fileloc,filename=None,*_):

        TerminalCost.__init__(self,boundaryconditions,fileloc,*_)
        
        if self.d > 1:
            raise NotImplementedError

        #initialise blank histogram values
        self.H = None
        self.yedges = None
        self.xedges = None 

    def update_cost_params(self,integrator,epoch):
        """make histogram of the final distribution"""
        self.update_M(epoch)

        x = integrator.update_ref_statistics().cpu().numpy()
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
        Estimate probability density at a point from the histogram.

        Parameters
        ----------
        datapoint : array-like, shape (2,)
            Data point or array of points.

        Returns
        -------
        p1 : float
            Estimate of probability density from histogram.
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
        self.update_M(epoch)

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
        make_scatter_plot(plot_loc,x.cpu().numpy(),self.rho.model,epoch)#,self.boundary)
        #distribution_plot(plot_loc,self.rho.model,self.boundary.params["Tf"],epoch,self.boundary)

    def get_logpdf(self,x,*_):
        pdf = self.rho.model.log_prob(x)
        return torch.nan_to_num(pdf)     