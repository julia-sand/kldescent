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

    def __init__(self,boundary,num_iters=None,num_samples=None):
        """
        class for solving the cell problem, which is a 'halfbridge' problem

        args: 
            boundary : class
                class storing the initial and final conditions of the problem
            num_iters : int
                how many iterations to perform
            num_samples : int 
                how many samples to use in each integration
        """
        
        self.boundary = boundary
        self.Tf2 = self.boundary.params["Tf"]*(self.boundary.params["epsilon"]**2)
        
        self.q = np.linspace(-4,4) #for plotting 

        self.A = self.boundary.afun(self.boundary.params["Tf"])
        self.B = self.boundary.bfun(self.boundary.params["Tf"])
        self.beta = 1/self.A

        self.power = self.B/(self.A+self.B)

        self.refsol = Gaussian_Overdamped_Ref()
        
        if num_samples is not None:
            self.num_samples = num_samples
        else:
            self.num_samples = 400
        if num_iters is not None:
            self.num_iters = num_iters
        else:
            self.num_iters = 3

        self.xsamples = np.linspace(-5,5,self.num_samples*100)

    def iteration(self):
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
        
        xhatphi = np.linspace(-5,5,num=self.num_samples)
        #self.boundary.generate_samples_q(self.num_samples, mode="final").flatten()
        
        #preallocate arrays
        hatphi0 = np.zeros_like(xhatphi) #these must be non.zero positive values
        hatphiT = np.exp(-1)*np.ones_like(xhatphi)
        psi0 = np.ones_like(xhatphi) #np.empty_like(xpsi) 
        psiT = np.ones_like(xhatphi) 
        
        xpsi = xhatphi.copy() #np.zeros_like(xhatphi)
        xtemp = xhatphi.copy()

        iters = 0
        while iters < self.num_iters:

            #compute psi0/phiT
            psi0 = self.boundary.q_marginal_final(xhatphi)/hatphiT#**(self.B/(self.A+self.B)) 

            #integrate psi to Tf
            xpsi,psiT = proximal_fpk(xhatphi,psi0+1e-10,self.Tf2,self.beta)

            #xpsi,psiT = self.new_iterate(xpsi_old,psiT_old)

            #compute hatphi0 
            hatphi0 = self.boundary.q_marginal_initial(xpsi)/psiT

            #integrate phihat to Tf
            xhatphi,hatphiT = proximal_fpk(xpsi,hatphi0+1e-10,self.Tf2,self.beta)
            #xhatphi,hatphiT = self.new_iterate(xhatphi_old,hatphiT_old)
            
            #xhatphi = xhatphi_temp 
            #xpsi = xpsi_temp 
            iters += 1

            self.plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
            self.plot_factors_PHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
            self.plot_factors_HATPHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

        return xhatphi,xpsi,hatphiT,psiT


    def resample_x(self,x,phi):
    
        ps_temp = interpolate_(self.xsamples,x,phi)
        return np.random.choice(self.xsamples,size=self.num_samples,p=ps_temp/np.sum(ps_temp))

    def new_iterate(self,xpsi_old,psiT_old):

        xpsi = self.resample_x(xpsi_old,psiT_old)
        psiT = interpolate_(xpsi,xpsi_old,psiT_old)
        plt.scatter(xpsi,psiT)
        plt.plot(self.q,self.refsol.phi_t(self.q,0))
        plt.savefig("temp.png")
        plt.close()
        return xpsi,psiT 

    def plot_factors_HATPHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):

        plt.scatter(xhatphi,hatphiT, label="t=T")
        #plt.scatter(xpsi,hatphi0, label="t=0")

        plt.plot(self.q,normalise(self.q,self.refsol.phi_t(self.q,self.Tf2)),linestyle="dashed",color="blue",alpha=0.5)
        plt.plot(self.q,normalise(self.q,self.refsol.phi_t(self.q,0)),linestyle="dashed",color="blue",alpha=0.5)

        #plt.xlim((-2,2))
        #plt.ylim((0,5))
        plt.legend()
        plt.savefig(f"factors0_hatphi_{iters}.png")
        plt.close()

    def plot_factors_PHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        
        #plt.scatter(xhatphi,psi0, label="t=T",zorder=100)
        plt.scatter(xpsi,normalise(xpsi,psiT), label="t=0")

        #plt.plot(self.q,self.refsol.phi_t(self.q,self.Tf2),linestyle="dashed",color="blue",alpha=0.5)
        plt.plot(self.q,normalise(self.q,self.refsol.hatphi_t(self.q,0)),linestyle="dashed",color="orange",alpha=0.5)
        #plt.xlim((-2,2))
        #plt.ylim((0,5))

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

    cell = CellProblem(boundary,num_iters=30)
    
    xhatphi,xpsi,hatphiT,psiT = cell.iteration()
