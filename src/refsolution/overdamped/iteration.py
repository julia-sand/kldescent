import numpy as np

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
                num_iters=1):
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
    
    x0 = boundary.generate_samples_q(num_samples, mode="initial").flatten()
    rho0 = boundary.q_marginal_initial(x0)
    rhoT = boundary.q_marginal_final(x0)
    
    Tf = boundary.params["Tf"]
    #preallocate arrays
    hatphi0 = np.empty_like(x0) 
    hatphiT = np.empty_like(x0) 
    psi0 = np.empty_like(x0) 
    psiT = np.exp(-1)*np.ones_like(x0) #these must be non.zero positive values

    beta = 1/A

    iters = 0
    while iters <= num_iters:
        xinit = x0.copy()
        #compute hatphi0
        hatphi0 = boundary.q_marginal_initial(xinit)/psiT

        #integrate phihat to Tf
        x0,hatphiT = proximal.proximal_fpk(xinit,hatphi0,Tf,beta)
        
        xnew = x0.copy()
        
        #compute psi0
        psi0 = (boundary.q_marginal_final(xnew)+1e-10/hatphiT)**(B/(A+B)) #(rhoT/normalise(x0,hatphiT))**(B/(A+B))

        #integrate psi to Tf
        x0,psiT = proximal.proximal_fpk(xnew,psi0,Tf,beta)

        iters += 1

    return x0,hatphiT,psiT


if __name__=="__main__":
    boundary = boundaries.UnderdampedGaussian()
    Tf = boundary.params["Tf"]

    x0,hatphiT,psiT = iteration(boundary,afun(Tf),bfun(Tf))

    import matplotlib.pyplot as plt 

    plt.scatter(x0,boundary.q_marginal_final(x0))
    plt.scatter(x0,hatphiT, label="phihat")
    plt.scatter(x0,psiT, label="psi")
    plt.legend()
    plt.savefig("iter_test1.png")
    plt.close()