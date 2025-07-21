"""
Calculates estimate of the probability density using method of 2411.08158

"""
import torch

import langevin 

class FokkerPlanck(langevin.SDE):
    
    def __init__(self):
        __super__.__init__(self) #use same intialisation

    def compute_joint(x0,net,
                        x_samples):

        net.eval()
        
        t_vec = np.array(self.boundary.get_reference_timegrid(t+.0001))
        
        #reset girsanov
        girsanov = torch.zeros((self.sde_params["batchsamples"],1))

        for i in range(1,len(t_vec)):#range(0,len(t_vec)):

            dt = np.abs(t_vec[-i-1]-t_vec[-i])
            
            dw = self.generate_innovation(dt) #np.sqrt(dt)*torch.randn((testing_samples,x_samples*x_samples,1))
            
            nn_input = torch.dstack((t_vec[-i]*torch.ones((testing_samples,x_samples**2,1),dtype=torch.float32),
                                        xinit[...,[0]]))
            du = torch.hstack([net(nn_input[:,i,:]) for i in range(x_samples**2)])[...,np.newaxis]

            #underdamped dynamics
            #backward (T-t0) evolution in t0
            xinit = boundary_cond.girsanov_scheme(xinit,du,-dt,-dw,t_vec[i])
            
            #remove escaped particles
            #xinit[xinit>30] = torch.nan
            #xinit[xinit<-30] = torch.nan

            #compute girsanov factor
            girsanov = girsanov + (np.sqrt(1/boundary_cond.sigma0)*(xinit[...,[1]]*dw) + (dt/(boundary_cond.sigma0**2))*torch.square(xinit[...,[1]]))

        return np.nanmean(np.multiply(boundary_cond.p_initial(xinit[...,[0]],xinit[...,[1]]),np.exp(-girsanov)),axis=0)

    