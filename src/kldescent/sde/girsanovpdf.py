import torch

from kldescent.utils.parser import integrator_params
from functorch import vmap

import pdb

@torch.no_grad
def girsanov_scheme(xinit,du_bar,dt,dw,t,sigma0,epsilon):

    #takes one step forward in the euler maruyama discretisation
    #go forward one step 
    xinit[...,[0]] = xinit[...,[0]] + epsilon*xinit[...,[1]]*dt
    xinit[...,[1]] = xinit[...,[1]] + sigma0*du_bar*dt + sigma0*dw

    return xinit

@torch.no_grad
def compute_pdf(x0,
                    net,
                    sigma0,
                    epsilon,
                    p_initial,
                    Tf,
                    mc_samples=None,
                    tsteps=None):
    """
    Compute an estimate of the final joint distribution using Girsanov's theorem.

    Gradient tracking is disabled.

    Based on: arXiv:2411.08518 (see Section 3 for an example application).

    Parameters
    ----------
    xinit : np.ndarray
        Coordinates (q, p) of x₀ at which to evaluate the density rho.

    net : object
        Neural network used for the computation.

    Tf : float
        Final time at which the distribution is evaluated.

    sigma0 : float
        Initial diffusion parameter of the dynamics.

    epsilon : float
        Small parameter governing the perturbative regime.

    pinitial : callable
        Function specifying the initial boundary condition.

    mcsamples : int
        Number of Monte Carlo samples used for the estimate.

    tsteps : int
        Number of time steps in the simulation.

    Returns
    -------
    pdfout : np.ndarray
        Estimated probability density function evaluated at time Tf and location xinit.
    """
    
    #pdb.set_trace()
    net.eval()
    
    #parse arguments if these not given
    if tsteps is None or mc_samples is None:
        args = integrator_params()

    #choose truthy values
    tsteps = tsteps or args.get("tsteps")
    mc_samples = mc_samples or args.get("testsamples")
    

    print("tsteps",tsteps)
    print("mc",mc_samples)

    with torch.no_grad():
        
        if mc_samples == 1:
            xinit = x0.clone().to(net.device)
        else:
            xinit =  x0.unsqueeze(0).expand(mc_samples, -1, -1).clone().to(net.device)
            #torch.broadcast_to(x0,(mc_samples,2)).clone().to(net.device)

        t_vec = torch.linspace(0,Tf,tsteps,requires_grad=False,dtype=torch.float32).to(net.device)

        #preallocate girsanov integrals and results
        girs = torch.zeros_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(net.device)
        pdfout = torch.zeros_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(net.device)

        for i in range(1,len(t_vec)):#self.sde_params["tsteps"]+1):
                            
            #get timestep
            dt = torch.abs(t_vec[-i-1]-t_vec[-i])
            
            dw = (dt**0.5)*torch.randn_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(net.device)

            #evaluate neural network approximated drift 
            du_bar = net(torch.cat((t_vec[-i]*torch.ones_like(xinit[...,[0]],dtype=torch.float32).to(net.device)
                                                        ,xinit[...,[0]]),dim=-1))
            
            #one backward step of euler maruyama scheme of stochastic evolution
            xinit = girsanov_scheme(xinit,du_bar,-dt,-dw,t_vec[-i],
                                                                sigma0,
                                                                epsilon,)
            
            #add step to stochastic integral
            girs = girs + (dw*xinit[...,[1]])/(2**0.5) + (dt * (xinit[...,[1]])**2)/4
                        
        pdfout = p_initial(xinit[...,[0]],xinit[...,[1]])*torch.exp(-girs)

    if pdfout.dim() > 2:
        return torch.mean(pdfout, dim=0)
    else:
        return pdfout

def plot_scatter_(x,pdf):
    plt.scatter(x[...,0].cpu().numpy(), x[...,1].cpu().numpy(), c=pdf.cpu().numpy(),cmap='RdYlGn_r')
    plt.title(f"{pdf.cpu().numpy().max()}")
    randint= random.randint(1, 100000) 
    plt.savefig(f"test_tests1.png")
    plt.close()

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import random
    from kldescent.sde.boundary import UnderdampedLandauer
    from kldescent.model.nn import NeuralNetwork

    boundary = UnderdampedLandauer()
    nn_temp = NeuralNetwork(boundary)
    testing_samples = 1000
    with torch.no_grad():
        xf = boundary.generate_samples(testing_samples,mode="final")

        pdf = torch.zeros_like(xf[...,[0]]).to(nn_temp.device)
        pdf = compute_pdf(xf,
                            nn_temp,
                            boundary.sigma0,
                            boundary.params["epsilon"],
                            boundary.p_initial,
                            boundary.params["Tf"],
                            mc_samples=2,
                            tsteps=21)#torch.vmap(lambda xi: torch.mean(), randomness='different')(xf) 
                
    plot_scatter_(xf, pdf)

