import numpy as np
import matplotlib.pyplot as plt 

from .proximal import proximal_fpk

def interpolate_(q,xs,phis):
    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx])

def normalise(x,y):
    idx = np.argsort(x)
    return y/np.trapz(y[idx],x[idx])

class CellProblem():

    def __init__(self,boundary,
                            A,
                            B):
        """
        class for solving the cell problem, which is a 'halfbridge' problem

        args: 
            boundary : class
                class storing the initial and final conditions of the problem
            A : float
                constant defining the cell problem
            B : float 
                constant defining the cell problem
        """
        
        self.boundary = boundary
        self.Tf2 = self.boundary.params["Tf"]*(self.boundary.params["epsilon"]**2)
        
        self.q = np.linspace(-4,4) #for plotting 

        self.A = A
        self.B = B
        self.beta = 1/A

    def iteration(self,num_samples=1000,
                    num_iters=3):
        """
        Implementation of Iteration in 1912.01244
        'Wasserstein Proximal Algorithms for the Schr¨odinger Bridge Problem:
        Density Control with Nonlinear Drift', Caluya&Halder,2021.
        
        args:
            num_samples : int
                number of samples
            num_iters : int
                number of iterations to perform

        returns
            hatphiT : arr
            psiT : arr
        """
        
        xhatphi = self.boundary.generate_samples_q(num_samples, mode="final").flatten()
        
        #preallocate arrays
        hatphi0 = np.empty_like(xhatphi) #these must be non.zero positive values
        hatphiT = np.exp(-1)*np.ones_like(xhatphi)
        psi0 = np.ones_like(xhatphi) #np.empty_like(xpsi) 
        psiT = np.ones_like(xhatphi) 
        
        iters = 0
        while iters <= num_iters:

            #compute psi0/phiT
            psi0 = (self.boundary.q_marginal_final(xhatphi)/normalise(xhatphi,hatphiT))#**(self.B/(self.A+self.B)) 

            #integrate psi to Tf
            xpsi,psiT = proximal_fpk(xhatphi,psi0,self.Tf2,self.beta)

            #compute hatphi0
            hatphi0 = self.boundary.q_marginal_initial(xpsi)/normalise(xpsi,psiT)

            #integrate phihat to Tf
            xhatphi,hatphiT = proximal_fpk(xpsi,hatphi0,self.Tf2,self.beta)
                
            iters += 1

            self.plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

            self.plot_factors(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

        return xhatphi,xpsi,hatphiT,psiT

    def plot_factors(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        
        plt.scatter(xpsi,psiT, label="phi0")
        plt.scatter(xhatphi,hatphiT, label="hatphi0")

        #plt.plot(self.q,self.boundary.q_marginal_final(self.q)**(self.B/(self.B+self.A)),linestyle="dashed",color="gray",alpha=0.5)
        #plt.plot(self.q,self.boundary.q_marginal_initial(self.q),linestyle="dashed",color="gray",alpha=0.5)

        plt.legend()
        plt.savefig(f"factors_iter_{iters}.png")
        plt.close()

    def plot_boundary(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        rhoT = interpolate_(self.q,xpsi,hatphiT)*interpolate_(self.q,xhatphi,psi0)
        rho0 =interpolate_(self.q,xpsi,hatphi0*psiT) #interpolate_(self.q,xhatphi,hatphi0)*interpolate_(self.q,xpsi,psiT)

        plt.plot(self.q,normalise(self.q,(rhoT)), label="rhoT")
        plt.plot(self.q,normalise(self.q,rho0), label="rho0")

        plt.plot(self.q,self.boundary.q_marginal_final(self.q)**(self.B/(self.B+self.A)),linestyle="dashed",color="gray",alpha=0.5)
        plt.plot(self.q,self.boundary.q_marginal_initial(self.q),linestyle="dashed",color="gray",alpha=0.5)

        plt.legend()
        plt.savefig(f"boundary_test_iter_{iters}.png")
        plt.close()

if __name__=="__main__":
    import src.sde.boundary as boundaries

    boundary = boundaries.UnderdampedGaussian()
    Tf = boundary.params["Tf"]
    Tf2 = boundary.params["Tf"]*(boundary.params["epsilon"]**2)
    
    #A = sol.afun(Tf)
    #B = sol.bfun(Tf)

    xhatphi,xpsi,hatphiT,psiT = iteration(boundary,1,1,Tf2)

    #plt.scatter(x0,boundary.q_marginal_final(x0))
    plt.scatter(xhatphi,hatphiT, label="phihat")
    plt.scatter(xpsi,psiT, label="psi")
    plt.legend()
    plt.savefig("iter_test1.png")
    plt.close()