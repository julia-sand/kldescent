import numpy as np
import matplotlib.pyplot as plt 

from ..heatintegrators import proximal_fpk,mcintegrator
from .gauss import *
from ..utilfunctions import *

class CellProblem():

    def __init__(self,boundary,save_location,bridge_type="marginal half"):
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
        self.save_location = save_location #where to save the plots
        
        self.Tf2 = self.boundary.params["Tf"]*(self.boundary.params["epsilon"]**2)
        
        self.A = self.boundary.afun(self.boundary.params["Tf"])
        self.B = self.boundary.bfun(self.boundary.params["Tf"])
        self.beta = 1/np.sqrt(self.A)
        self.M = self.boundary.params["weight"]


        if bridge_type=="marginal half":
            self.power = (self.M*self.B)/(self.A+self.M*self.B)
        elif bridge_type=="full":
            self.power = 1
        else:
            self.power = 1
            print("Bridge type not found. Default values used.")

        #only initialise when using gaussian boundaries
        if "Gauss" in str(self.boundary.__class__.__name__):
            self.refsol = Gaussian_Overdamped_Ref(boundary)
            self.qmax = 15
        else:            
            self.qmax = 6
        
        self.q = np.linspace(-self.qmax,self.qmax) #for plotting 


        #preallocate the arrays here 
        self.xhatphi = None #np.linspace(-6,6,num_samples)
        self.hatphiT = None # np.ones_like(xhatphi)
        self.psiT = None #np.ones_like(xhatphi) 
        self.hatphi0 = None # np.ones_like(xhatphi)
        self.psi0 = None #np.ones_like(xhatphi) 
        self.xpsi = None

    def iteration(self,integrator,num_iters=4,num_samples=400):
        """
        Implementation of Iteration in 1912.01244
        'Wasserstein Proximal Algorithms for the Schr¨odinger Bridge Problem:
        Density Control with Nonlinear Drift', Caluya&Halder,2021.
        
        args:
            integrator : method
                method to use for integrating the heat equations
            num_samples : int
                number of samples
            num_iters : int
                number of iterations to perform

        returns
            hatphiT : arr
            psiT : arr
        """
        
        xhatphi = np.linspace(-self.qmax,self.qmax,num_samples)
        #self.boundary.generate_samples_q(num_samples, mode="final").flatten()
        
        #preallocate arrays
        hatphi0 = np.ones_like(xhatphi) #these must be non.zero positive values
        hatphiT = np.ones_like(xhatphi)
        psi0 = np.ones_like(xhatphi) #np.empty_like(xpsi) 
        psiT = np.ones_like(xhatphi) 
        
        xpsi = xhatphi.copy() #np.zeros_like(xhatphi)

        iters = 0
        while iters < num_iters:

            #compute psi0/phiT
            psi0 = np.nan_to_num(zchop(self.boundary.q_marginal_final(xhatphi)/hatphiT),posinf=0,neginf=0,nan=0)**self.power

            #integrate psi to Tf
            xpsi,psiT = integrator(xhatphi,psi0,self.Tf2,self.beta)

            #compute hatphi0 
            hatphi0 = np.nan_to_num(zchop(self.boundary.q_marginal_initial(xpsi)/psiT),posinf=0,neginf=0,nan=0)

            #integrate phihat to Tf
            xhatphi,hatphiT = integrator(xhatphi,hatphi0,self.Tf2,self.beta)
            
            iters += 1

            self.plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
            
            self.plot_factors_PHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
            self.plot_factors_HATPHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

        #set new class properties
        self.xhatphi =xhatphi
        self.xpsi = xpsi 
        self.hatphiT = hatphiT
        self.psiT = psiT
        self.hatphi0 = hatphi0
        self.psi0 = psi0

        return 

    def plot_factors_HATPHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):

        #plt.scatter(xhatphi,normalise(xhatphi,hatphiT), label="t=T")
        #plt.scatter(xhatphi,hatphi0, label="t=0")
        plt.scatter(xpsi,hatphiT, label="t=T")
        try:
            plt.plot(self.q,normalise(self.q,self.refsol.hatphi_t(self.q,self.Tf2)),linestyle="dashed",color="blue",alpha=0.5)
            plt.plot(self.q,normalise(self.q,self.refsol.hatphi_t(self.q,0)),linestyle="dashed",color="blue",alpha=0.5)
        except:
            pass

        plt.legend()
        plt.savefig(self.save_location+f"/factors0_hatphi_{iters}.png")
        plt.close()

    def plot_factors_PHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        
        plt.scatter(xhatphi,psi0, label="t=0",zorder=100)
        plt.scatter(xpsi,psiT, label="t=T",zorder=100)
        #plt.scatter(xpsi,normalise(xpsi,psiT), label="t=0")

        try:
            plt.plot(self.q,normalise(self.q,self.refsol.phi_t(self.q,self.Tf2)),linestyle="dashed",color="blue",alpha=0.5)
            plt.plot(self.q,normalise(self.q,self.refsol.phi_t(self.q,0)),linestyle="dashed",color="orange",alpha=0.5)
        except:
            pass  

        plt.legend()
        plt.savefig(self.save_location+f"/factors_phi_{iters}.png")
        plt.close()

    def plot_boundary(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        rhoT = interpolate_(self.q,xpsi,hatphiT)*interpolate_(self.q,xhatphi,psi0)
        rho0 = interpolate_(self.q,xpsi,hatphi0)*interpolate_(self.q,xhatphi,psiT) #interpolate_(self.q,xhatphi,hatphi0)*interpolate_(self.q,xpsi,psiT)
        
        #plt.plot(self.q,normalise(self.q,rhoT)**self.power)
        plt.plot(self.q,normalise(self.q,rhoT), label="rhoT")
        plt.plot(self.q,normalise(self.q,rho0), label="rho0")

        plt.plot(self.q,self.boundary.q_marginal_final(self.q),linestyle="dashed",color="gray",alpha=0.5)
        plt.plot(self.q,self.boundary.q_marginal_initial(self.q),linestyle="dashed",color="gray",alpha=0.5)

        plt.legend()
        plt.savefig(self.save_location+f"/boundary_test_iter_{iters}.png")
        plt.close()

if __name__=="__main__":
    import src.sde.boundary as boundaries

    boundary = boundaries.UnderdampedLandauer()
    Tf = boundary.params["Tf"]

    cell = CellProblem(boundary)
    
    cell.iteration(mcintegrator,num_iters=4)
