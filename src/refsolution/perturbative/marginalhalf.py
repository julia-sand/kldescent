import numpy as np

from scipy.ndimage import median_filter,generic_filter
import scipy.ndimage as sc

import pdb


class MarginalHalfBridgePerturbation():
    """Class to contain the functions for computing perturbative 
    corrections from overdamped data for the Marginal Half Bridge;
    that is, when the terminal cost is specified by the KL divergence between the 
    realised final distribution and a reference one (the logarithm of their ratio)

    args:
        boundary : class
            determines boundary conditions and parameters
    """

    def __init__(self,boundary,reference):

        self.boundary = boundary
        self.reference = reference

        #load the csv
        self.overdamped_arr = self.reference.get_ref_data(data="overdamped")

        #define parameters
        self.t2vec = np.unique(self.overdamped_arr[:,0])
        self.xnum = int(self.overdamped_arr.shape[0]/len(self.t2vec))
        self.Tf = self.boundary.params["Tf"]
        self.epsilon = self.boundary.params["epsilon"]
        self.Tf2 = self.Tf*(self.epsilon**2)
        
        self.qaxis,_ = self.reference.get_quantity(0,"value",arr=self.overdamped_arr)
        
        self.filter_delta = 40 
        
    def apply_filter(self,y,mode="nearest"):

        return generic_filter(y,sc.median,size=self.filter_delta,mode=mode)

    def compute_gradient(self,y,x):
 
        return np.gradient(y,x)#self.apply_filter(np.gradient(y,x),mode)

    def get_t0(self,t2):

        return round(t2/(self.epsilon**2),4)

    def v00(self,t2):

        _,v00 = self.reference.get_quantity(t2,"value",arr=self.overdamped_arr)
        return v00 

    def f00(self,t2):

        _,f00 = self.reference.get_quantity(t2,"marginal",arr=self.overdamped_arr)
        return f00

    def v11(self,t0,t2):
        
        dv00_temp = self.compute_gradient(self.v00(t2),self.qaxis)
        coeff = 1- np.exp(t0-self.Tf)

        return coeff*dv00_temp

    def DU(self,t0,t2):
        """Finds the perturbative drift for the marginal half bridge
        args:
            t0 : float
                fast time
            t2 : float
                slow time
        """
        
        return 2*self.v11(t0,t2)
    
    def f02(self,t0,t2):
        #pdb.set_trace()
                
        v00_temp = self.v00(t2)
        dv00_temp = self.apply_filter(self.compute_gradient(v00_temp,self.qaxis),mode="nearest")
        del v00_temp

        f00_temp = self.apply_filter(self.f00(t2),mode="constant")
        ddv00_temp = self.apply_filter(self.compute_gradient(dv00_temp,self.qaxis),mode="nearest")
        prod2 = f00_temp*ddv00_temp

        del ddv00_temp 
        
        df00_temp = self.apply_filter(self.compute_gradient(f00_temp,self.qaxis),mode="constant")
        del f00_temp 

        prod1 = df00_temp*dv00_temp
        term1 = -t0*df00_temp
        del dv00_temp
        
        ddf00_temp = self.apply_filter(self.compute_gradient(df00_temp,self.qaxis),mode="constant")
        del df00_temp

        term2 = ddf00_temp*(t0+np.exp(-t0)-1)
        del ddf00_temp 

        term3 = -2*(np.sinh(t0)-t0-(np.cosh(t0)-1)*(1-np.exp(-self.Tf)))*(prod1 + prod2)    
            
        return term1 + term2 + term3

    def marginal_perturbative(self,t0,t2):
        
        return self.f00(t2) + (self.epsilon**2)*self.f02(t0,t2)

    def xcorr(self,t0,t2):
        return 0

    def momvar(self,t0,t2):
        return 0

    def posvar(self,t0,t2):
        return 0

    def mommean(self,t0,t2):
        return 0

    def posmean(self,t0,t2):
        return 0

    def save_perturbative(self,):
        """
        compute perturbative drift and adds to 
        csv and saves
        """

        pertsol = np.c_[self.overdamped_arr, np.ones(self.xnum*len(self.t2vec)), np.ones(self.xnum*len(self.t2vec))] 
        
        for t in range(len(self.t2vec)):

            #perturbative drift 
            pertsol[t*self.xnum:(t+1)*self.xnum,-1] = self.DU(self.get_t0(self.t2vec[t]),self.t2vec[t])
            
            #perturbative marginal
            pertsol[t*self.xnum:(t+1)*self.xnum,-2] = self.marginal_perturbative(self.get_t0(self.t2vec[t]),self.t2vec[t])

        #save pertsol 
        np.savetxt(self.reference.get_ref_path()+"/perturbative_underdamped.csv",pertsol,fmt="%10.6f",header = "t,q,rho,V,f,DU",delimiter=',')
        #self.reference.add_perturbative_underdamped_data(self.reference.get_ref_path()+"/perturbative_underdamped.csv")

    def save_perturbative_cumulants(self,):
        """
        computes estiamtes of the perturbative cumulants and saves as a csv
        """

        #preallocate
        pertcumulants = np.empty((len(self.t2vec),6))
        
        for t in range(len(self.t2vec)):
            
            pertcumulants[t,0] = self.get_t0(self.t2vec[t])
            pertcumulants[t,1] = self.posmean(self.get_t0(self.t2vec[t]),self.t2vec[t])
            pertcumulants[t,2] = self.mommean(self.get_t0(self.t2vec[t]),self.t2vec[t])
            pertcumulants[t,3] = self.posvar(self.get_t0(self.t2vec[t]),self.t2vec[t])
            pertcumulants[t,4] = self.momvar(self.get_t0(self.t2vec[t]),self.t2vec[t])
            pertcumulants[t,5] = self.xcorr(self.get_t0(self.t2vec[t]),self.t2vec[t])

        #save pertsol 
        np.savetxt(self.reference.get_ref_path()+"/cumulants.csv",pertcumulants,fmt="%10.6f",header = "t0,posmean,mommean,posvar,momvar,xcorr",delimiter=',')

if __name__=="__main__":
    from ..reference import RefSolution
    from ...sde.boundary import *
    
    boundary = UnderdampedLandauer()
    reference = RefSolution(boundary=boundary)
    pertsol = MarginalHalfBridgePerturbation(boundary,reference)
    pertsol.save_perturbative()
    pertsol.save_perturbative_cumulants()