import torch

from ..utils.parser import integrator_params

@torch.no_grad
def girsanov_scheme(xinit,du_bar,dt,dw,t,sigma0,epsilon):

    #takes one step forward in the euler maruyama discretisation
    #go forward one step 
    xinit[...,[1]] = xinit[...,[1]] + sigma0*du_bar*dt + sigma0*dw
    xinit[...,[0]] = xinit[...,[0]] + epsilon*xinit[...,[1]]*dt

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
    Computes an estimate for the final joint distribution using Girsanov theorem 
    Does not track gradients 
    Following 2411.08518 (see Section 3 for example). 

    ====
    Parameters
        xinit : array
            coordinate (q,p) of x0 on which to evaluate rho
        net : class
            neural net to use 
        Tf : float 
            time for the plot
        sigma0 : float
            parameter of the evolution
        epsilon : float
            parameter of the evolution 
        pinitial : method
            the initial condition of the boundary conditions
        mcsamples : int
            number of mc samples
        tsteps : int
            number of timesteps
    Returns: 
        pdfout : arr
            estimate of the pdf at time Tf at location xinit
    ====
    """
    
    net.eval()

    #todo fix this
    args = integrator_params()
    tsteps = args["tsteps"]
    if mc_samples is None: #use default if none given
        mc_samples = args["testsamples"]

    with torch.no_grad():
        
        if mc_samples == 1:
            xinit = x0.clone().to(net.device)
        else:
            xinit = torch.broadcast_to(x0,(mc_samples,2)).clone().to(net.device)

        t_vec = torch.linspace(0,Tf,tsteps,requires_grad=False,dtype=torch.float32).to(net.device)

        #preallocate girsanov integrals and results
        girs = torch.zeros_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(net.device)
        pdfout = torch.zeros_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(net.device)

        for i in range(1,len(t_vec)):#self.sde_params["tsteps"]+1):
                            
            #get timestep
            dt = torch.abs(t_vec[-i-1]-t_vec[-i])
            
            dw = (dt**0.5)*torch.randn_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32).to(net.device)

            #evaluate neural network approximated drift 
            du_bar = net(torch.cat((t_vec[-i]*torch.ones_like(xinit,dtype=torch.float32).to(net.device)
                                                        ,xinit[...,[0]]),dim=-1))
            
            #one backward step of euler maruyama scheme of stochastic evolution
            xinit = girsanov_scheme(xinit,du_bar,-dt,-dw,t_vec[-i],
                                                                sigma0,
                                                                epsilon,)
            
            #add step to stochastic integral
            girs = girs + (dw*xinit[...,[1]])/(2**0.5) + (dt * (xinit[...,[1]])**2)/4
                        
        pdfout = p_initial(xinit[...,[0]],xinit[...,[1]])*torch.exp(-girs)

    return pdfout
