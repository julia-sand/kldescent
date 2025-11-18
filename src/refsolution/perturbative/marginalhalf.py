import numpy as np

from .reference import *
from ..sde.boundary import *

#contains perturbative calculations for marginal halfbridge
class MarginalHalfBridgePerturbation():
    """Class to contain the functions for computing perturbative 
    corrections from overdamped data
    args:
        boundary : class
            determines boundary conditions and parameters
    """

    def __init__(self,boundary):

        self.boundary = boundary
        self.reference_ = RefSolution(boundary)  
        self.Tf = self.boundary.params["Tf"]
        self.epsilon = self.boundary.params["epsilon"]
        self.Tf2 = self.Tf*(self.epsilon**2)

    def dv00(self,t2):

        x,v00 = self.reference_.get_overdamped_quantity(t2,"value")
        return np.gradient(v00,x) 

    def v11(self,t0):
        
        coeff = 1- np.exp(t0-self.Tf)

        t2 = t0*(self.epsilon**2)
        return coeff*dv00(t2)

    def DU(self,t0):
        """Finds the perturbative drift for the marginal half bridge
        """

        return 2*v11(t0,self.Tf)
    
if __name__=="__main__":
    boundary = UnderdampedLandauer()

    x,f00 = reference_.get_overdamped_quantity(time,"rho")
