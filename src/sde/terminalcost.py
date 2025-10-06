"""This file contains methods for estimating the final value function 
in the half bridge. 
Including 
norm-flow:
parametric: When we know the final distribution is Gaussian, we can use a parametric method of distirbution estimation, namely estimate first and second moments by MLE. 
rho-net: Use girsanov theorem to evaluate 
hist: Use a histogram to estimate the final value function
"""

import numpy as np 


class TerminalCost():
    """Class for methods for estimating the final value function 
    """

    def __init__(self,boundaryconditions):
        """
        Initialise how the terminal cost should be evaluated.    
        args:
            boundaryconditions: 
                associated boundary conditions
        """

        self.boundary = boundaryconditions

    def marginal_halfbridge(self,):
        """
        Terminal cost is the ratio of the final estimated configuration distribution divided by the log of the 
        reference configuration distribution
        g(q,p) = log(^P_marg(q,p) / Pref_marg)
        """

        pass

    def quarterbridge(self,):
        pass

    def full_halfbridge(self,):
        """
        Terminal cost is the ratio of the final estimated state distribution divided by the log of the 
        reference state distribution
        g(q,p) = log(^P(q,p) / Pref)
        """
        
        pass

class GirsanovNet(TerminalCost):
    """
    Estimate the final cost with a neural network trained on the final distribution
    """

    def __init__(self,boundaryconditions,rhonet):
        """
        args:
            rhonet: class
                Neural network that can be used to estimate the final disitrbution
        """

        TerminalCost().__init__(boundaryconditions)
        self.rhonet=rhonet

    def gfun(self,x):
        pdf = self.rhonet(x,condition="log")
        return np.nan_to_num(pdf) + (self.boundary.Ufinal(x[...,[0]])+((x[...,[1]]**2)/2))

class MLE(TerminalCost):
    """
    Estimate the final value function using an estimate of the distribution
    """

    def __init__(self,rhonet):
        """
        args:
            rhonet: class
                Neural network that can be used to estimate the final disitrbution
        """

        TerminalCost().__init__()
        self.rhonet=rhonet

    def gfun(self,x):
        return np.nan_to_num(pdf) + (self.Ufinal(x[...,[0]])+((x[...,[1]]**2)/2))
