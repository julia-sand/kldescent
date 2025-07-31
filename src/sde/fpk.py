"""
Calculates estimate of the joint probability density using method of 2411.08158

"""
import torch
import numpy as np

import pdb

from ..sde.langevin import SDE 
from ..utils.parser import fetch_results_folder_from_cmd
from ..sde import boundary as boundaries 
from ..utils.getboundary import *

class FokkerPlanck(SDE):
    """
    Integrator class for integrating the Fokker planck equation
    Most arguments are the same as the Langevin integrator, except:

    args: 
        net: 
            neural net used in the integration that approximates the drift
        
    """
    
    def __init__(self,net,boundary_cond,
                    sde_params=None,
                    logger=None,
                    device="cpu"):

        SDE.__init__(self,boundary_cond,sde_params,logger) #use same intialisation
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
        
        self.net.eval()
    
        with torch.no_grad():
            t_vec = self.generate_time_grid()

            #set up a,abar,c,cbar
            girs = torch.zeros((self.sde_params["batchsamples"],1),requires_grad=False,dtype=torch.float32)

            for i in range(0,self.sde_params["tsteps"]-1):
                
                dt = (t_vec[i+1]-t_vec[i])
                
                dw = self.generate_innovation(dt,self.sde_params["batchsamples"])
            
                du_bar = self.boundary.du_ref(xinit[...,[1]],t_vec[i])
                #self.net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
                        #                                   ,xinit[...,[0]])))
                
                girs = girs + (((2*dt)**(-0.5))*(xinit[...,[1]])) + ((dt * (xinit[...,[1]])**2)/4)

                xinit = self.boundary.girsanov_scheme(xinit,du_bar,-dt,-dw,t_vec[i])
            
            pdfout = torch.mean(self.boundary.p_initial(xinit[...,[0]],xinit[...,[1]])*torch.exp(-girs))

        return pdfout
    
    def loss_fn(self,rhonet):

        #initialise x values 
        x0 = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),dtype=torch.float32)#.to_device(device)
        pdfout = self.compute_pdf(x0.detach())

        return torch.mse(rhonet(x0),pdfout)


if __name__=="__main__":
    import os 

    from ..model.nn import NeuralNetwork
    from ..sde.boundary import UnderdampedHalfGaussian
    
    folder = fetch_results_folder_from_cmd()
    
    boundary_cond, data_loaded = init_params_from_file(folder)

    net = NeuralNetwork(boundary_cond,data_loaded)

    fpk = FokkerPlanck(net,boundary_cond)
    x0 = torch.tensor(fpk.boundary.generate_samples(fpk.sde_params["batchsamples"],"initial"),dtype=torch.float32)#.to_device(device)

    pdfout = fpk.compute_pdf(x0.detach())
    print(np.mean(pdfout.numpy() - fpk.boundary.p_final(x0[...,[0]],x0[...,[1]]).numpy()))