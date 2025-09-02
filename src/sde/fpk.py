"""
Calculates estimate of the joint probability density using method of 2411.08158

"""
import math

import torch
import numpy as np

import pdb

from ..sde.langevin import SDE 
from ..utils.parser import fetch_results_folder_from_cmd
from ..sde import boundary as boundaries 
from ..utils.getboundary import *

from ..viz.contour import *


class FokkerPlanck(SDE):
    """
    Integrator class for integrating the Fokker planck equation
    Most arguments are the same as the Langevin integrator, except:

    args: 
        net: 
            neural net used in the integration that approximates the drift
        
    """
    
    def __init__(self,boundary_cond,net,
                    sde_params=None,
                    device="cpu"):

        SDE.__init__(self,boundary_cond,
                            sde_params=sde_params) #use same intialisation
        self.net = net 

    def compute_pdf(self,xinit):
        """
        Following 2411.08518 (see Section 3 for example). 

        ====
        Parameters
            xinit: values of x0 on which to evaluate rho
        Returns: 
            pdfout: estimate of the pdf at final time at locations xinit
        ====
        """
        #xinit = x0.clone()

        num_samples = xinit.shape[0]
    
        with torch.no_grad():
            t_vec = self.generate_time_grid()

            girs = torch.zeros((num_samples,1),requires_grad=False,dtype=torch.float32)

            for i in range(1,self.sde_params["tsteps"]+1):
                #print(t_vec[-i])
                #print(t_vec[-i-1])
                #print("====")
                
                dt = np.abs(t_vec[-i-1]-t_vec[-i])
                
                dw = self.generate_innovation(dt,num_samples)
            
                du_bar = self.net(torch.column_stack((t_vec[-i]*torch.ones((num_samples,1),dtype=torch.float32)
                                                           ,xinit[...,[0]])))
                #(0.2/(2**0.5))*self.net(torch.column_stack((t_vec[i]*torch.ones((num_samples,1),dtype=torch.float32)
                #                                           ,xinit[...,[0]])))
                

                xinit = self.boundary.girsanov_scheme(xinit,du_bar,-dt,-dw,t_vec[-i])
                girs = girs + (dw*xinit[...,[1]])/(2**0.5) + (dt * (xinit[...,[1]])**2)/4
               
                #girs + (dw*xinit[...,[1]])/(self.boundary.sigma0) + (dt * (xinit[...,[1]])**2)/4
            
            pdfout = self.boundary.p_initial(xinit[...,[0]],xinit[...,[1]])*torch.exp(-girs)

        return pdfout
    
    def loss_fn(self,rhonet,x0_clone):
        """
        Computes pdf values and returns l2 error loss function 
        
        args: 
            rhonet:
                neural network to be trained
        """

        #x0 = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"))#.to_device(device)
        with torch.no_grad():
            pdfout = self.compute_pdf(x0_clone)
        
        return pdfout
        
if __name__=="__main__":
    
    import os 

    from ..model.nn import NeuralNetwork,RhoNeuralNetwork
    from ..sde.boundary import UnderdampedHalfGaussian
    
    boundary_cond = UnderdampedHalfGaussian()

    net = NeuralNetwork(boundary_cond)
    rhonet = RhoNeuralNetwork(boundary_cond)

    fpk = FokkerPlanck(boundary_cond,net)
    
    fpk.loss_fn(rhonet)
