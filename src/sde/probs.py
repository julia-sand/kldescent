"""
Calculates estimate of the joint probability density using method of 2411.08158

"""
import torch
import numpy as np

import pdb

from ..sde.langevin import SDE 

class FokkerPlanck(SDE):
    
    def __init__(self,boundary_cond,
                    sde_params=None,
                    logger=None,
                    device="cpu"):

        SDE.__init__(self,boundary_cond,sde_params,logger) #use same intialisation

    def compute_pdf(self,net):
        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural network
        ====

        """
        
        net.eval()
        t_vec = self.generate_time_grid()

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)

        #set up a,abar,c,cbar
        girs = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)

        with torch.no_grad():
            t_vec = self.generate_time_grid()

            #initialise x values 
            xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)

            #set up a,abar,c,cbar
            girs = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)

            for i in range(0,self.sde_params["tsteps"]-1):
                
                dt = (t_vec[i+1]-t_vec[i])
                
                dw = self.generate_innovation(dt,self.sde_params["batchsamples"])
            
                du_bar = self.boundary.du_ref(xinit[...,[1]],t_vec[i])
                #net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
                        #                                   ,xinit[...,[0]])))
                
                girs = girs + (((2*dt)**(-0.5))*(xinit[...,[1]])) + ((dt * (xinit[...,[1]])**2)/4)

                xinit = self.boundary.girsanov_scheme(xinit,du_bar,-dt,-dw,t_vec[i])
            
            pdfout = torch.mean(self.boundary.p_initial(xinit[...,[0]],xinit[...,[1]])*torch.exp(-girs))
                        
        
        return xinit,pdfout
    
    def check_mean(self,xinit,pdfout):
        return np.trapz(np.trapz(pdfout.to_numpy(),np.sort(xinit[...,1].to_numpy())),xinit[...,0].to_numpy())


if __name__=="__main__":
    import os 

    from ..model.nn import NeuralNetwork
    from ..sde.boundary import UnderdampedHalfGaussian
    
    boundary_cond = UnderdampedHalfGaussian()
    net = NeuralNetwork(boundary_cond)

    fpk = FokkerPlanck(boundary_cond)
    x,pdfout = fpk.compute_pdf(net)
    print(fpk.check_mean(x,pdfout))