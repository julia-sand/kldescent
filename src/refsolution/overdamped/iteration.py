import warnings #ignore divide by zeros for the boundaries

import numpy as np
import matplotlib.pyplot as plt 

from ..heatintegrators import proximal_fpk,mcintegrator
from .gauss import *
from ..utilfunctions import *

class CellProblemBase():
    """Base class for setting parameters for cell problem"""
    
    def __init__(self,boundary,save_location):

        self.boundary = boundary
        self.save_location = save_location #where to save the plots
        
        self.Tf = self.boundary.params["Tf"]
        self.Tf2 = self.boundary.params["Tf"]*(self.boundary.params["epsilon"]**2)
        
        self.M = self.boundary.params["weight"]
        
        self.qmax = self.boundary.qmax
        
        self.q = np.linspace(-self.qmax,self.qmax,num_samples) #for plotting 

class CellProblem(CellProblemBase):

    def __init__(self,boundary,save_location,bridge_type="marginal"):
        """
        class for solving the cell problem, which is a 'marginal halfbridge' problem

        args: 
            boundary : class
                class storing the initial and final conditions of the problem
            num_iters : int
                how many iterations to perform
            num_samples : int 
                how many samples to use in each integration
            bridge_type : str
        """
        
        CellProblemBase.__init__(boundary,save_location)
        
        self.A = self.afun(self.boundary.params["Tf"],bridge_type)
        self.B = self.bfun(self.boundary.params["Tf"],bridge_type)
        self.beta = 1/np.sqrt(self.A)

        #####################################
        
        if bridge_type =="marginal":
                         
            self.power = (self.M*self.B)/(self.A+self.M*self.B)

        elif bridge_type == "full":

            #to be updated 
            self.beta = 1/np.sqrt(self.A)
        
            self.power = 1
        else:
            
            self.power = 1
            print("Bridge type not found. Default value used.")

        #only initialise when using gaussian boundaries
        if "Gauss" in str(self.boundary.__class__.__name__):
            self.refsol = Gaussian_Overdamped_Ref(boundary)
            #
        else:            
            pass #self.qmax = 6

        self.xhatphi = np.linspace(-self.qmax,self.qmax,num_samples)
        #self.boundary.generate_samples_q(num_samples, mode="final").flatten()
        
        #preallocate arrays
        self.hatphi0 = np.ones_like(xhatphi) #these must be non.zero positive values
        self.hatphiT = np.ones_like(xhatphi)
        self.psi0 = np.ones_like(xhatphi) #np.empty_like(xpsi) 
        self.psiT = np.ones_like(xhatphi) 
        self.xpsi = np.ones_like(xhatphi)

    def bfun(self,Tf,bridge_type):
        
        if bridge_type=="marginal":
            return 1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf)
        else:
            return 1 

    def afun(self,Tf,bridge_type):
       
        if bridge_type=="marginal":
            return (Tf - 1 + np.exp(-Tf))/Tf
        else:
            return 1 

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

        #initialise iterates
        xhatphi = self.xhatphi
        hatphi0 = self.hatphi0 #these must be non.zero positive values
        hatphiT = self.hatphiT
        psi0 = self.psi0  #np.empty_like(xpsi) 
        psiT = self.psiT 
        xpsi = self.xpsi

        iters = 0
        
        with warnings.catch_warnings(): #dont print divide by zero warnings (these are handled in the script)
            warnings.simplefilter("ignore")
            
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

        #plot the final output
        self.plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
        
        self.plot_factors_PHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
        self.plot_factors_HATPHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

        #set new iterates
        self.xhatphi = zchop(xhatphi)
        self.xpsi = zchop(xpsi) 
        self.hatphiT = zchop(hatphiT)
        self.psiT = zchop(psiT)
        self.hatphi0 = zchop(hatphi0)
        self.psi0 = zchop(psi0)

        return 

    def plot_factors_HATPHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):

        plt.scatter(xpsi,hatphiT, label="t=T")
        try: #plot known solution for gaussians
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

        try: #plot known solution for gaussians
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

class HalfBridgeCellProblem(CellProblemBase):
    """Separate class for the half bridge cell problem, because it is 
    different to the other cases
    """
    def __init__(self,boundary,):
        CellProblemBase.__init__(self,boundary,save_location)

        self.A = self.afun(self.Tf)
        self.B = self.bfun(self.Tf)

        self.alpha = 0.5*(np.sqrt(A)*(1-2*self.M)-1)

    def bfun(self,Tf):
        
        return -(np.tanh(Tf/2)/Tf)*((1-np.exp(-Tf))**2)/(1+4*np.exp(-Tf)+np.exp(-2*Tf))
        
    def afun(self,Tf):
       
        return -(-1 + (2*np.tanh(Tf/2)/Tf)-self.bfun(Tf,bridge_type="half"))
    
    def constant_hboundary(self,q,varkappa,Tf):
        """Compute the constant for the value function boundary condition
        """

        return  -self.B*((varkappa/(2*self.A))*q + (1-self.A)*Tf*(varkappa**2)/4)

    def varkappa(self,rho,sigma):
        """Compute the constant varkappa
        for the boundary conditions
        
        args:
            rho : ndarray
                computed by phi times hatphi 
            sigma : ndarray
                computed by log phi 
        returns 
            varkappa
        """
        
        coeff = 2*self.A/(self.A-self.B)

        out = rho*np.gradient(sigma,self.qaxis)
        return np.trapezoid(out,self.qaxis)

    
     
    def iteration(self,integrator,num_iters=4,num_samples=400):
        """
        Adaptation of the Iteration in 1912.01244
        'Wasserstein Proximal Algorithms for the Schr¨odinger Bridge Problem:
        Density Control with Nonlinear Drift', Caluya&Halder,2021.
        
        args:
            integrator : method
                method to use for integrating the heat equations. Since we have an extra 
                term on the boundary, we need to use a discretised version  
            num_samples : int
                number of samples
            num_iters : int
                number of iterations to perform

        returns
            hatphiT : arr
            psiT : arr
        """

        #initialise iterates
        xhatphi = self.xhatphi
        hatphi0 = self.hatphi0 #these must be non.zero positive values
        hatphiT = self.hatphiT
        psi0 = self.psi0  #np.empty_like(xpsi) 
        psiT = self.psiT 
        xpsi = self.xpsi

        iters = 0
        
        with warnings.catch_warnings(): #dont print divide by zero warnings (these are handled in the script)
            warnings.simplefilter("ignore")
            
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

        #plot the final output
        self.plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
        
        self.plot_factors_PHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
        self.plot_factors_HATPHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

        #set new iterates
        self.xhatphi = zchop(xhatphi)
        self.xpsi = zchop(xpsi) 
        self.hatphiT = zchop(hatphiT)
        self.psiT = zchop(psiT)
        self.hatphi0 = zchop(hatphi0)
        self.psi0 = zchop(psi0)

        return 
   
if __name__=="__main__":
    import src.sde.boundary as boundaries

    boundary = boundaries.UnderdampedLandauer()
    Tf = boundary.params["Tf"]

    cell = CellProblem(boundary)
    
    cell.iteration(mcintegrator,num_iters=4)
