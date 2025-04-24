import numpy as np
import numpy.random as npr
import sde.boundary as boundary
import torch

#samples from the initial / final joint density
#we use the fact that the position/momentum is independent on the boundaries
#for faster sampling


#generate the initial data
def generate_samples_q(batch_samples,mode):
    
    ##sample points from the intial distribution for q 
    #output as a tensor

    #from where to sample the q marginals
    if mode=="initial":
        q_marginal = q_marginal_initial
    elif mode=="final": 
        q_marginal = q_marginal_final

    probs = q_marginal(np.linspace(-8,8,10000))/np.sum(q_marginal(np.linspace(-8,8,10000)))
    
    
    return torch.tensor(npr.choice(np.linspace(-8,8,10000),
                                    batch_samples,
                                    p=probs),dtype=torch.float32).reshape((batch_samples,1))
    #torch.tensor(np.column_stack((npr.choice(np.linspace(-8,8,10000),
    #                                batch_samples,
    #                                p=probs), npr.randn(batch_samples))),dtype=torch.float32)



#generate the initial data
def generate_samples(batch_samples,mode):
    
    ##sample points from the intial distribution for q 
    ##and from normal distribution for p and stack together 
    #output as a tensor

    #gives the initialisation of the underdamped

    
    
    return torch.column_stack((generate_samples_q(batch_samples,mode)
                            ,torch.randn((batch_samples,1),dtype=torch.float32)))
    
    #torch.tensor(np.column_stack((npr.choice(np.linspace(-8,8,10000),
    #                                batch_samples,
    #                                p=probs), npr.randn(batch_samples))),dtype=torch.float32)


if __name__=="__main__":
    
    #enerate_samples(100,"initial")
    pass