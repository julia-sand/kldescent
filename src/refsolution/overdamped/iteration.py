import numpy as np
import matplotlib.pyplot as plt 

import src.sde.boundary as boundaries
import proximal,sol

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

    def iteration(self,num_samples=200,
                    num_iters=4):
        """Implementation of Iteration in 1912.01244
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
        
        xpsi = self.boundary.generate_samples_q(num_samples, mode="initial").flatten()
        
        
        #preallocate arrays
        hatphi0 = np.empty_like(xpsi) 
        hatphiT = np.empty_like(xpsi) 
        psi0 = np.empty_like(xpsi) 
        psiT = np.exp(-1)*np.ones_like(xpsi) #these must be non.zero positive values
    
        #beta = 1/A

        iters = 0
        while iters <= num_iters:
            
            #compute hatphi0
            hatphi0 = boundary.q_marginal_initial(xpsi)/psiT

            #integrate phihat to Tf
            xhatphi,hatphiT = proximal.proximal_fpk(xpsi,hatphi0,self.Tf2,self.beta)
                
            #compute psi0
            psi0 = (boundary.q_marginal_final(xhatphi)/hatphiT)**(self.B/(self.A+self.B)) #(rhoT/normalise(x0,hatphiT))**(B/(A+B))

            #integrate psi to Tf
            xpsi,psiT = proximal.proximal_fpk(xhatphi,psi0,self.Tf2,self.beta)

            iters += 1
            self.plot_boundary(self.boundary,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

        return xhatphi,xpsi,hatphiT,psiT

    def plot_boundary(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        
        rhoT = sol.interpolate_(self.q,xpsi,hatphiT)*sol.interpolate_(q,xhatphi,psi0)
        rho0 = sol.interpolate_(self.q,xhatphi,hatphi0)*sol.interpolate_(q,xpsi,psiT)

        plt.plot(self.q,sol.normalise(self.q,(rhoT)), label="rhoT")
        plt.plot(self.q,sol.normalise(self.q,rho0), label="rho0")

        plt.plot(self.q,self.boundary.q_marginal_final(q)**(self.B/(self.B+self.A)),linestyle="dashed",color="gray",alpha=0.5)
        plt.plot(self.q,self.boundary.q_marginal_initial(q),linestyle="dashed",color="gray",alpha=0.5)

        plt.legend()
        plt.savefig(f"boundary_test_iter_{iters}.png")
        plt.close()

if __name__=="__main__":
    boundary = boundaries.UnderdampedGaussian()
    Tf = boundary.params["Tf"]
    Tf2 = boundary.params["Tf"]*(boundary.params["epsilon"]**2)
    
    A = sol.afun(Tf)
    B = sol.bfun(Tf)

    xhatphi,xpsi,hatphiT,psiT = iteration(boundary,A,B,Tf2)

    #plt.scatter(x0,boundary.q_marginal_final(x0))
    plt.scatter(xhatphi,hatphiT, label="phihat")
    plt.scatter(xpsi,psiT, label="psi")
    plt.legend()
    plt.savefig("iter_test1.png")
    plt.close()