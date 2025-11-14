import numpy as np
import scipy.spatial.distance as sci

import pdb


def mcintegrator(x0,y0,
                    Tf,
                    beta,
                    num_steps = 10,
                    mc_samples = 1000):
    """
    Integrates a heat equation by the monte carlo method
    yT = E(Phi(xT)|x0)
    where xT are final positions of trajectories of the auxiliary dynamics
    Parameters:
        x0 : 
            initial positions, where the pde will be evaluated 
        y0 : arr
            initial values at x0, used to interpolate the intial function
        Tf : float
            final time t
        num_tsteps : float
            number of timesteps
        mcsamples : int
            number of MonteCarlo trajectories
    Returns
        yT : array
            values at final time evaluated with feynman-kac
    """

    Generator = np.random.default_rng(seed=42)
    
    num_qsamples = x0.shape[0]
    #pdb.set_trace()
    qax = x0.copy()
    
    def phifinal(qT,x0,y0):
        idx = np.argsort(x0)
        return np.interp(qT,x0[idx],y0[idx],left=0 ,right=0)

    t_vec = np.linspace(0,Tf,num_steps) #time vector
    tstep = t_vec[1] #size of time interval
    
    #generate trajectories
    qtest = np.tile(qax[:,np.newaxis],(1,mc_samples))
    traj =  qtest + \
        np.sqrt(2*tstep/beta)*np.sum(Generator.standard_normal((num_qsamples,mc_samples,num_steps)),axis=2)
    #traj = np.flip(traj,axis = 1) #flip along columns
    yT = np.mean(phifinal(traj,qax,y0),axis=1) 
    return x0, yT

def proximal_fpk(x0,y0,
                    Tf,
                    beta,
                    num_steps=200, 
                    max_iters=1000, 
                    eps=0.05, 
                    tol=1e-12): 
    """Implementation of algorithm described in 
    1908.00533 "Gradient Flow Algorithms for Density Propagation in Stochastic Systems"; Caluya&Halder,2019.
    
    Integrates a simple heat equation model 
    args:
        x0 : arr
            array size (n,) of samples from starting distribution
        y0 : arr
            array size (n,) of probabilities of each sample x0
        Tf : float
            time up to which to integrate
        beta : float
            coefficient of laplacian in heat equation
        num_steps : int
            number of timesteps
        max_iters : int
            maximum number of sinkhorn iterations per timestep
        eps : float
            sinkhorn regularisation constant
        tol : float
            tolerance at which to stop iteration at each timestep
    
    returns: 
        x_end : arr
            array of shape (n,) of final positions (unordered)
        y_end : arr
            evaluated solution (unnormalised) at each point in x_end
    """
    
    #set up sampler
    Generator = np.random.default_rng(seed=42)
    
    samples = x0.shape[0] #number of points in the cloud
    x_evolution = x0.copy()

    dt = Tf/(num_steps-1)

    #preallocate arrays
    x_evolution_new = np.empty_like(x_evolution) #np.zeros((samples,num_steps))

    power = 1/(1+(beta*eps/dt))
    yvals = y0.copy()#/np.sum(y0)

    for k in range(num_steps):
        
        #step of sde evolution 
        innovation = np.sqrt(2*dt/beta)*Generator.standard_normal(size=x_evolution.shape)

        x_evolution_new = x_evolution + innovation
        c_k = sci.cdist(x_evolution_new[:,np.newaxis],x_evolution[:,np.newaxis],metric="sqeuclidean")#

        gamma_k = np.exp(-c_k/(2*eps))
        #pdb.set_trace()

        gamma_kt = gamma_k.T 

        z_samples = Generator.standard_normal(size=x_evolution.shape)

        z_old = np.exp((dt*z_samples)/eps) 
        y_old = np.divide(yvals,gamma_k@z_old)

        #preallocate arrays
        y_new = np.empty_like(y_old)
        z_new = np.empty_like(y_old)

        xi = np.exp(-1)*np.ones_like(x_evolution)

        iters = 0 

        while iters<=max_iters: 
            z_new = np.divide(xi,gamma_kt@y_old)**power
            y_new = np.divide(yvals,gamma_k@z_new) 

            if ((np.sum(np.abs(y_new - y_old))<tol) and
                    (np.sum(np.abs(z_new - z_old))<tol)):
                
                #update iterates
                y_old = y_new
                z_old = z_new
                break
            
            else: 
                iters += 1
            
            #update iterates
            y_old = y_new
            z_old = z_new
        
        #set new running solutions
        yvals = np.multiply(z_new,gamma_kt@y_new)
        x_evolution = x_evolution_new

    return x_evolution,yvals

if __name__=="__main__":
    import matplotlib.pyplot as plt 

    import src.sde.boundary as boundaries
    from src.refsolution.marginal_overdamped.gauss import Gaussian_Overdamped_Ref,normalise
    Generator = np.random.default_rng(seed=42)
    boundary = boundaries.UnderdampedGaussian()
    refsol = Gaussian_Overdamped_Ref(boundary)
    Tf2 = boundary.params["Tf"]*(boundary.params["epsilon"]**2)
    
    beta = 1/(boundary.afun(boundary.params["Tf"]))
    nsamples = 500

    x0 = boundary.generate_samples_q(nsamples,"final").flatten()
    y0 = refsol.phi_t(x0,Tf2)
    
    x0_mc = np.linspace(-4,4,nsamples)

    #x_end, y_end = proximal_fpk(x0,y0,Tf2,beta,num_steps=1000)
    x_end_mc, y_end_mc = mcintegrator(x0_mc,refsol.phi_t(x0_mc,Tf2),Tf2,beta,num_steps=100)

    #plt.scatter(x_end, normalise(x_end,y_end),label="prox")#normalise(x_end,y_end))
    plt.plot(x_end_mc, normalise(x_end_mc,y_end_mc),label="mc")#normalise(x_end,y_end))
    
    qax = np.linspace(-4,4)
    plt.plot(qax,normalise(qax,refsol.phi_t(qax,Tf2)),label="phi at Tf")
    plt.plot(qax,normalise(qax,refsol.phi_t(qax,0)),label="phi at 0")
    plt.plot(qax,normalise(qax,refsol.hatphi_t(qax,Tf2)),label="hatphi at Tf")
    plt.plot(qax,normalise(qax,refsol.hatphi_t(qax,0)),label="hatphi at 0")
    
    plt.legend()
    #plt.ylim((0,2))
    plt.savefig("test_fpk_phi_5.png")
    plt.close()