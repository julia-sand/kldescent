import numpy as np
import matplotlib.pyplot as plt 

from .proximal import proximal_fpk
from .gauss import *

def interpolate_(q,xs,phis):
    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx])

def normalise(x,y):
    idx = np.argsort(x)
    return y.copy()/np.trapz(y[idx],x[idx])

def sig_initial(x,y):
    idx = np.argsort(x)
    ynorm = normalise(x,y)

    return np.trapz(ynorm[idx]*x[idx]**2,x[idx])
    

class CellProblem():

    def __init__(self,boundary):
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

        self.A = self.boundary.afun(self.boundary.params["Tf"])
        self.B = self.boundary.bfun(self.boundary.params["Tf"])
        self.beta = 1/self.A

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
        
        xhatphi = np.linspace(-0.5,0.5,num_samples)#self.boundary.generate_samples_q(num_samples, mode="final").flatten()
        
        #preallocate arrays
        hatphi0 = np.empty_like(xhatphi) #these must be non.zero positive values
        hatphiT = np.exp(-1)*np.ones_like(xhatphi)
        psi0 = np.ones_like(xhatphi) #np.empty_like(xpsi) 
        psiT = np.ones_like(xhatphi) 
        xpsi = np.zeros_like(xhatphi)
        
        iters = 0
        while iters <= num_iters:

            #compute psi0/phiT
            psi0 = phi_t(xhatphi,self.Tf2)
            #self.boundary.q_marginal_final(xhatphi)/hatphiT#normalise(xhatphi,hatphiT))#**(self.B/(self.A+self.B)) 

            #integrate psi to Tf
            xpsi_temp,psiT = proximal_fpk(xhatphi,psi0,self.Tf2,self.beta)

            #compute hatphi0 
            hatphi0 = hatphi_t(xpsi,0)
            idx_temp = np.random.choice(num_samples,p=hatphi0/np.sum(hatphi0),size=num_samples)
            #self.boundary.q_marginal_initial(xpsi)/psiT#normalise(xpsi,psiT)

            #integrate phihat to Tf
            xhatphi_temp,hatphiT = proximal_fpk(xpsi[idx_temp],hatphi0[idx_temp],self.Tf2,self.beta)
            
            xhatphi = xhatphi_temp 
            xpsi = xpsi_temp 
            iters += 1

            #self.plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

            self.plot_factors_PHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
            self.plot_factors_HATPHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

        return xhatphi,xpsi,hatphiT,psiT

    def plot_factors_HATPHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        plt.scatter(xhatphi,hatphiT, label="t=T")
        #plt.scatter(xpsi,hatphi0, label="t=0")

        plt.plot(self.q,hatphi_t(self.q,self.Tf2),linestyle="dashed",color="blue",alpha=0.5)
        #plt.plot(self.q,hatphi_t(self.q,0),linestyle="dashed",color="orange",alpha=0.5)

        plt.xlim((-2,2))
        plt.ylim((0,5))
        plt.legend()
        plt.savefig(f"factors0_hatphi_{iters}.png")
        plt.close()

    def plot_factors_PHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        
        #plt.scatter(xhatphi,psi0, label="t=T",zorder=100)
        plt.scatter(xpsi,psiT, label="t=0")

        plt.plot(self.q,phi_t(self.q,self.Tf2),linestyle="dashed",color="blue",alpha=0.5)
        plt.plot(self.q,phi_t(self.q,0),linestyle="dashed",color="orange",alpha=0.5)
        plt.xlim((-2,2))
        plt.ylim((0,5))

        plt.legend()
        plt.savefig(f"factors_phi_{iters}.png")
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

    cell = CellProblem(boundary)
    
    xhatphi,xpsi,hatphiT,psiT = cell.iteration(num_iters=1)
