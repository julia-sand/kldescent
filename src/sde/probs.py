"""
Calculates estimate of the joint probability density using method of 2411.08158

"""
import torch

import pdb

from ..sde.langevin import SDE 

class FokkerPlanck(SDE):
    
    def __init__(self,boundary_cond,
                    sde_params=None,
                    logger=None,
                    device="cpu"):

        SDE.__init__(self,boundary_cond,sde_params,logger) #use same intialisation

    def simulation_free(self,net):

        """
        Following 2410.05163 (see proposition 3). 

        ====
        Parameters
        - net: neural network
        ====

        """
        
        net.train()
        t_vec = self.generate_time_grid()

        #initialise x values 
        xinit = torch.tensor(self.boundary.generate_samples(self.sde_params["batchsamples"],"initial"),requires_grad=False,dtype=torch.float32)#.to_device(device)

        #set up a,abar,c,cbar
        a = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        abar = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        c = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        varc = torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
        girs = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)
        j = torch.zeros((self.sde_params["batchsamples"],1),dtype=torch.float32)

        with torch.no_grad():
            varc *= xinit[...,[1]]**2

        #pdb.set_trace()
        #self.generate_time_grid()
        for i in range(0,self.sde_params["tsteps"]-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = self.generate_innovation(dt,self.sde_params["batchsamples"])
           
            du = net(torch.column_stack((t_vec[i]*torch.ones((self.sde_params["batchsamples"],1),dtype=torch.float32)
                                                    ,xinit[...,[0]])))
            a = a + dt*(torch.linalg.norm(du, axis=1, keepdim=True)*xinit[...,[1]]) 
       
            c = c + (du*dw)

            du_bar = du.detach() # . #untracked copy

            with torch.no_grad(): #no gradient 

                abar = abar + dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True)*xinit[...,[1]]) 
                #cbar = c.detach()
                #abar = abar - dt*(torch.linalg.norm(du_bar, axis=1, keepdim=True)*xinit[...,[1]]) 
                #bbar = bbar + dt*(self.boundary.f(xinit,t_vec[i])) 
                girs = girs + (dw * (2**(-0.5))*(xinit[...,[1]])) + ((dt * (xinit[...,[1]])**2)/4)

                xinit = self.boundary.girsanov_scheme(xinit,du_bar,dt,dw,t_vec[i])

        with torch.no_grad():
            #multiply by the constants
            varc *= (2**0.5)/(4*self.boundary.params["epsilon"])
            #varc *= xinit[...,[1]]**2
            varc *= self.boundary.Uinitial(xinit[...,[0]])#,xinit[...,[1]]))
            varc *= girs
            #varc = varc + torch.log(self.boundary.p_initial(xinit[...,[0]],xinit[...,[1]]))
            j = varc*abar + torch.log(self.boundary.p_final(xinit[...,[0]],xinit[...,[1]])) 
        
        #compute value of Lhat
        #see equatio§n (10) and (11) in ref. 2410.05163
        #Lhat = torch.mean(0.5 * a + ((0.5 * abar + bbar + self.boundary.gfun(xinit)) * c))
        #Lhat = torch.mean(0.5 * a + (((1/4) * abar + bbar + self.boundary.gfun(xinit)) * c))
        Phat = torch.mean(varc*a - j*c)
                    
        
        #free up memory again
        del xinit 
        del du
        del du_bar
        del t_vec
        del dw

        a = None
        c = None

        return Phat


if __name__=="__main__":
    import os 

    from ..model.nn import NeuralNetwork
    from ..sde.boundary import UnderdampedHalfGaussian
    
    boundary_cond = UnderdampedHalfGaussian()
    net = NeuralNetwork(boundary_cond)

    fpk = FokkerPlanck(boundary_cond)
    print(fpk.simulation_free(net))