import warnings #ignore divide by zeros for the boundaries

import numpy as np
import matplotlib.pyplot as plt 

from ..heatintegrators import proximal_fpk,mcintegrator
from .gauss import *
from ..utilfunctions import *

class CellProblem():

    def __init__(self,boundary,save_location,bridge_type):
        """
        class for solving the cell problem using an iteration algorithm

        args: 
            boundary : class
                class storing the initial and final conditions of the problem
            num_iters : int
                how many iterations to perform
            num_samples : int 
                how many samples to use in each integration
            bridge_type : str
        """
        
        self.bridge_type = bridge_type

        ##boundary parameters
        self.boundary = boundary
        self.save_location = save_location #where to save the plots
        
        self.Tf = self.boundary.params["Tf"]
        self.Tf2 = self.boundary.params["Tf"]*(self.boundary.params["epsilon"]**2)
        
        self.M = self.boundary.params["weight"]
        
        self.qmax = self.boundary.qmax
        self.num_samples=400
        self.q = np.linspace(-self.qmax,self.qmax,self.num_samples) #for plotting 

        ######################################
        #bridge type parameters
        self.B = self.bfun(self.boundary.params["Tf"])
        self.A = self.afun(self.boundary.params["Tf"])
        self.beta = 1/np.sqrt(self.A)

        self.power = self.compute_power()

        #####################################
        
        #only initialise when using gaussian boundaries
        if "Gauss" in str(self.boundary.__class__.__name__):
            self.refsol = Gaussian_Overdamped_Ref(boundary,self.A,self.B)
            #
        else:            
            pass #self.qmax = 6

        self.xhatphi = self.q 
        
        #preallocate arrays
        self.hatphi0 = np.ones_like(self.xhatphi) #these must be non.zero positive values
        self.hatphiT = np.ones_like(self.xhatphi)
        self.psi0 = np.ones_like(self.xhatphi) #np.empty_like(xpsi) 
        self.psiT = np.ones_like(self.xhatphi) 
        self.xpsi = self.xhatphi 

    def compute_power(self):
        if self.bridge_type =="marginal":
                         
            return (self.M*self.B)/(self.A+self.M*self.B)
        elif self.bridge_type == "full":
            #to be updated 
       
            return 1
        elif self.bridge_type == "half":
            self.alpha = ((1-np.sqrt(self.A))/(2*self.A))-self.M
            return np.sqrt(self.A)/(1-np.sqrt(self.A)*self.alpha)
        else:  
            print("Bridge type not found. Default value used.")
            return 1

    def bfun(self,Tf):
        
        if self.bridge_type=="marginal":
            return 1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf)
        elif self.bridge_type =="half":
            return -(np.tanh(Tf/2)/Tf)*((1-np.exp(-Tf))**2)/(1+4*np.exp(-Tf)+np.exp(-2*Tf))
        else:
            return 1 

    def afun(self,Tf):
       
        if self.bridge_type=="marginal":
            return (Tf - 1 + np.exp(-Tf))/Tf
        elif self.bridge_type =="half":
            return -(-1 + (2*np.tanh(Tf/2)/Tf)-self.B)  
        else:
            return 1 

    def compute_boundary_value(self,xhatphi,hatphiT,psi0,xpsi):
        if self.bridge_type == "marginal":
            return zchop(self.boundary.q_marginal_final(xhatphi)/hatphiT)
        elif self.bridge_type == "full":
            return zchop(self.boundary.q_marginal_final(xhatphi)/hatphiT)
        elif self.bridge_type == "half":
            #print(r"$\alpha$=",self.alpha)
            varkappa = self.compute_varkappa(hatphiT,psi0,xpsi)
            #print(r"$\kappa$=",varkappa)
            C = np.exp(self.constant_hboundary_exponent(xhatphi,varkappa))
            #print(r"$\gamma$=",C)
            return (hatphiT**self.alpha)*C*(self.boundary.q_marginal_final(xhatphi)**self.M)
        
    def constant_hboundary_exponent(self,q,varkappa):
        """Compute the constant for the value function boundary condition
        """

        return (self.B/(2*self.A))*(varkappa*q + (1-self.A)*self.Tf*(varkappa**2)/2)

    def compute_varkappa(self,hatphi,phi,xphi):
        """Compute the constant varkappa
        for the boundary conditions
        
        args:
            hatphi : ndarray
            phi : ndarray
            xpsi : ndarray
        returns 
            varkappa
        """
        
        coeff = -2*np.sqrt(self.A)/(self.A-self.B)
        out = hatphi*np.gradient(phi,xphi)
        return coeff*np.trapezoid(out,xphi)

    def l2err(self,x,y):
        return np.mean((x-y)**2)

    def iteration(self,integrator,
                        tol=1e-4,
                        max_iters=4):
        """
        Implementation of Iteration in 1912.01244
        'Wasserstein Proximal Algorithms for the Schr¨odinger Bridge Problem:
        Density Control with Nonlinear Drift', Caluya&Halder,2021.
        
        args:
            integrator : method
                method to use for integrating the heat equations
            max_iters : int
                maximum number of iterations to perform
            tol : float
                tolerance for difference between the iterations. If below tol, stop the iteration. 

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
            
            while iters < max_iters:

                #compute psi0/phiT
                psi0 = np.nan_to_num(self.compute_boundary_value(xhatphi,hatphiT,psi0,xpsi),posinf=0,neginf=0,nan=0)**self.power

                #integrate psi to Tf
                xpsi,psiT = integrator(xhatphi,psi0,self.Tf2,self.beta)

                #compute hatphi0 
                hatphi0 = np.nan_to_num(zchop(self.boundary.q_marginal_initial(xpsi)/psiT),posinf=0,neginf=0,nan=0)

                #integrate phihat to Tf
                xhatphi,hatphiT = integrator(xhatphi,hatphi0,self.Tf2,self.beta)
            
                #plot the final output
                self.plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
                
                self.plot_factors_PHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
                self.plot_factors_HATPHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

                if (self.l2err(hatphiT,self.hatphiT)<tol and self.l2err(psiT,self.psiT)<tol):
                    break
                else: 
                    iters += 1

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

    cell = CellProblem(boundary,"results_M20_v2","half")
    
    cell.iteration(mcintegrator)
