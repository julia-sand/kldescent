import numpy as np
import matplotlib.pyplot as plt 

import src.sde.boundary as boundaries
import proximal

#Proximal operator solution for fokkerplanck equation
def afun(Tf):
    return (Tf - 1 + np.exp(-Tf))/Tf

def bfun(Tf):
    return 1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf)

def normalise(x,y):
    idx = np.argsort(x)
    return y/np.trapz(y[idx],x[idx])

def iteration(boundary,
                A,
                B,
                num_samples=200,
                num_iters=4):
    """Implementation of Iteration in 1912.01244
    'Wasserstein Proximal Algorithms for the Schr¨odinger Bridge Problem:
    Density Control with Nonlinear Drift', Caluya&Halder,2021.
    
    to solve an overdamped 'halfbridge' problem

    args: 
        boundary : class
            class storing the initial and final conditions of the problem
        A : float
            constant defining the overdamped bridge problem 
        B : float  
            constant defining the overdamped bridge problem
        num_samples : int
            number of samples
        num_iters : int
            number of iterations to perform

    returns
        hatphiT : arr
        psiT : arr
    """
    
    xpsi = boundary.generate_samples_q(num_samples, mode="initial").flatten()
    #rho0 = boundary.q_marginal_initial(x0)
    #rhoT = boundary.q_marginal_final(x0)
    
    #compute slow time
    Tf = boundary.params["Tf"]*(boundary.params["epsilon"]**2)
    
    #preallocate arrays
    hatphi0 = np.empty_like(xpsi) 
    hatphiT = np.empty_like(xpsi) 
    psi0 = np.empty_like(xpsi) 
    psiT = np.exp(-1)*np.ones_like(xpsi) #these must be non.zero positive values

    beta = 1/A

    iters = 0
    while iters <= num_iters:
        
        #compute hatphi0
        hatphi0 = boundary.q_marginal_initial(xpsi)/psiT

        #integrate phihat to Tf
        xhatphi,hatphiT = proximal.proximal_fpk(xpsi,hatphi0,Tf,beta)
               
        #compute psi0
        psi0 = (boundary.q_marginal_final(xhatphi)/hatphiT)**(B/(A+B)) #(rhoT/normalise(x0,hatphiT))**(B/(A+B))

        #integrate psi to Tf
        xpsi,psiT = proximal.proximal_fpk(xhatphi,psi0,Tf,beta)

        iters += 1
        plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

    return xhatphi,xpsi,hatphiT,psiT

def interpolate_(q,x,phi):
    idx = np.argsort(x)
    return np.interp(q,x[idx],phi[idx])

def get_transient_factors(q,xpsi,psiT,t):
    """
    Evaluate the schrödinger factors along the time horizon 
    args: 
        t : float
            time to compute the factor
    returns:
        phi_t : arr
        hatphi_t : arr
    """

    #evolve hatphi0 to hatphi_t
    #get hatphi0 
    hatphi0 = boundary.q_marginal_initial(xpsi)/psiT
    #returns the factor at time t evolved by proximal operator
    xhatphi_t,hatphi_t = proximal.proximal_fpk(xpsi,hatphi0,t,1/A)

    #get phi_t = psi_(T-t)
    #returns the factor at time t evolved by proximal operator
    xphi_t,phi_t = proximal.proximal_fpk(xpsi,psiT,t,1/A)
    
    return interpolate_(q,xphi_t,phi_t),interpolate_(q,xhatphi_t,hatphi_t)

def plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
    q = np.linspace(-4,4,)
    
    rhoT = interpolate_(q,xpsi,hatphiT)*interpolate_(q,xhatphi,psi0)
    rho0 = interpolate_(q,xhatphi,hatphi0)*interpolate_(q,xpsi,psiT)

    plt.plot(q,normalise(q,rhoT), label="rho ref")
    plt.plot(q,normalise(q,rho0), label="rho0")

    plt.plot(q,boundary.q_marginal_final(q),linestyle="dashed",color="gray",alpha=0.5)
    plt.plot(q,boundary.q_marginal_initial(q),linestyle="dashed",color="gray",alpha=0.5)

    
    plt.legend()
    plt.savefig(f"boundary_test_iter_{iters}.png")
    plt.close()

if __name__=="__main__":
    boundary = boundaries.UnderdampedGaussian()
    Tf = boundary.params["Tf"]

    xhatphi,xpsi,hatphiT,psiT = iteration(boundary,afun(Tf),bfun(Tf))


    #plt.scatter(x0,boundary.q_marginal_final(x0))
    plt.scatter(xhatphi,hatphiT, label="phihat")
    plt.scatter(xpsi,psiT, label="psi")
    plt.legend()
    plt.savefig("iter_test1.png")
    plt.close()