import warnings #ignore divide by zeros for the boundaries

import numpy as np
import matplotlib.pyplot as plt 

from kldescent.benchmark.heatintegrators import proximal_fpk,mcintegrator
from kldescent.benchmark.overdamped.gauss import *
from kldescent.benchmark.utilfunctions import *
from kldescent.utils.parser import cell_argparser

class CellProblem():

    def __init__(self,boundary,
                      save_location,
                      bridge_type):
        """
        Class for solving the cell problem using an iterative algorithm.

        Parameters
        ----------
        boundary : object
            Object containing the boundary conditions of the problem.

        num_iters : int
            Number of iterations to perform in the solver.

        num_samples : int
            Number of samples used in each integration step.

        bridge_type : str
            Type of bridge formulation to use (e.g. "marginal", "half", "full").
        """
        
        self.bridge_type = bridge_type

        ##boundary parameters
        self.boundary = boundary
        self.save_location = save_location #where to save the plots
        
        self.Tf = self.boundary.params["Tf"]
        self.Tf2 = self.boundary.params["Tf"]*(self.boundary.params["epsilon"]**2)
        
        self.M = self.boundary.params["weight"]
        
        self.qmax = self.boundary.qmax
        
        cell_args = cell_argparser()
        self.num_samples = cell_args["cellxnum"]
        self.num_tsteps = cell_args["celltnum"]

        self.q = np.linspace(-self.qmax,self.qmax,self.num_samples)  

        ######################################
        #parameters dependent on bridge type
        self.B = bfun(self.bridge_type,self.Tf) #self.bfun()
        self.A = afun(self.bridge_type,self.Tf)
        self.beta = 1/np.sqrt(self.A)

        self.power = self.compute_power()

        #####################################
        
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
        
            return 1
        elif self.bridge_type == "half":
        
            self.alpha = ((1-np.sqrt(self.A))/(2*np.sqrt(self.A))) - self.M #((1-np.sqrt(self.A))/(2*self.A))-self.M
            return 1/(((1+np.sqrt(self.A))/(2*np.sqrt(self.A))) + self.M)
        else:  
            print("Bridge type not found. Default value used.")
            return 1

    def compute_boundary_value(self,xhatphi,hatphiT,psi0,xpsi):
        if self.bridge_type == "marginal":
            return np.nan_to_num(zchop(self.boundary.q_marginal_final_np(xhatphi)/hatphiT),posinf=0,neginf=0,nan=0)
        elif self.bridge_type == "full":
            return np.nan_to_num(zchop(self.boundary.q_marginal_final_np(xhatphi)/hatphiT),posinf=0,neginf=0,nan=0)
        elif self.bridge_type == "half":
            varkappa = self.compute_varkappa(hatphiT,psi0,xpsi)
            C = np.exp(self.constant_hboundary_exponent(xhatphi,varkappa))
            out = (hatphiT**self.alpha)*C*(self.boundary.q_marginal_final_np(xhatphi)**self.M)
            return np.nan_to_num(out,posinf=0,neginf=0,nan=0)
    
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

    def set_new_iterates(self,xhatphi,xpsi,hatphiT,psiT,hatphi0,psi0):
        
        self.xhatphi = zchop(np.nan_to_num(xhatphi,posinf=0,neginf=0,nan=0))
        self.xpsi = zchop(np.nan_to_num(xpsi,posinf=0,neginf=0,nan=0))
        self.hatphiT = zchop(np.nan_to_num(hatphiT,posinf=0,neginf=0,nan=0))
        self.psiT = zchop(np.nan_to_num(psiT,posinf=0,neginf=0,nan=0))
        self.hatphi0 = zchop(np.nan_to_num(hatphi0,posinf=0,neginf=0,nan=0))
        self.psi0 = zchop(np.nan_to_num(psi0,posinf=0,neginf=0,nan=0))
    
    def iteration(self,integrator,
                        tol=1e-5,
                        max_iters=10):
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
                equal to phi0
        """

        #initialise iterates
        xhatphi = self.xhatphi
        hatphi0 = self.hatphi0 #these must be non.zero positive values
        hatphiT = self.hatphiT
        psi0 = self.psi0  #np.empty_like(xpsi) 
        psiT = self.psiT 
        xpsi = self.xpsi

        iters = 0
        
        print("Starting iteration")
        with warnings.catch_warnings(): #dont print divide by zero warnings (these are handled in the script)
            warnings.simplefilter("ignore")
            
            while iters < max_iters:
                
                print("Iteration ",iters)
                #compute psi0/phiT
                psi0 = self.compute_boundary_value(xhatphi,hatphiT,psi0,xpsi)**self.power

                #normalise
                psi0 = normalise(xhatphi,psi0)

                #integrate psi to Tf
                xpsi,psiT = integrator(xhatphi,psi0,self.Tf2,self.beta)

                #compute hatphi0 
                hatphi0 = np.nan_to_num(zchop(self.boundary.q_marginal_initial_np(xpsi)/psiT),posinf=0,neginf=0,nan=0)
                hatphi0 = normalise(xpsi,hatphi0)

                #integrate phihat to Tf
                xhatphi,hatphiT = integrator(xhatphi,hatphi0,self.Tf2,self.beta)
                #print("Error:",self.l2err(psiT,self.psiT))
                if self.l2err(psi0,self.psi0)<tol:# and self.l2err(hatphiT,self.hatphiT)<tol):
                    self.set_new_iterates(xhatphi,xpsi,hatphiT,psiT,hatphi0,psi0)
                    break
                else: 
                    self.set_new_iterates(xhatphi,xpsi,hatphiT,psiT,hatphi0,psi0)
                    iters += 1

                #plot the final output
                self.plot_boundary(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
                
                self.plot_factors_PHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)
                self.plot_factors_HATPHI(xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters)

        print("Iteration complete.")
        return 

    def plot_factors_HATPHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):

        plt.scatter(self.xpsi,self.hatphiT, label="t=T")

        plt.legend()
        plt.savefig(self.save_location+f"/factors0_hatphi_{iters}.png")
        plt.close()

    def plot_factors_PHI(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        
        plt.scatter(self.xhatphi,self.psi0, label="t=0",zorder=100)
        plt.scatter(self.xpsi,self.psiT, label="t=T",zorder=100)

        plt.legend()
        plt.savefig(self.save_location+f"/factors_phi_{iters}.png")
        plt.close()

    def plot_boundary(self,xpsi,xhatphi,psi0,psiT,hatphi0,hatphiT,iters):
        
        rhoT = interpolate_(self.q,self.xpsi,self.hatphiT)*interpolate_(self.q,self.xhatphi,self.psi0)
        rho0 = interpolate_(self.q,self.xpsi,self.hatphi0)*interpolate_(self.q,self.xhatphi,self.psiT) #interpolate_(self.q,xhatphi,hatphi0)*interpolate_(self.q,xpsi,psiT)
        
        plt.plot(self.q,normalise(self.q,rhoT), label="rhoT")
        plt.plot(self.q,normalise(self.q,rho0), label="rho0")

        plt.plot(self.q,self.boundary.q_marginal_final_np(self.q),linestyle="dashed",color="gray",alpha=0.5)
        plt.plot(self.q,self.boundary.q_marginal_initial_np(self.q),linestyle="dashed",color="gray",alpha=0.5)

        plt.legend()
        plt.savefig(self.save_location+f"/boundary_test_iter_{iters}.png")
        plt.close()

    
if __name__=="__main__":
    import kldescent.sde.boundary as boundaries

    boundary = boundaries.UnderdampedLandauer()
    Tf = boundary.params["Tf"]

    cell = CellProblem(boundary,"results_M20_v2","half")
    
    cell.iteration(mcintegrator)
