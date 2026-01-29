import numpy as np

from scipy.ndimage import median_filter,generic_filter
import scipy.ndimage as sc

import pdb

class Perturbation():

    def __init__(self,boundary,overdamped_arr,ref_path):
        
        self.boundary = boundary
        
        self.ref_path = ref_path

        #load the csv
        self.overdamped_arr = overdamped_arr

        #define parameters
        self.t2vec = np.unique(self.overdamped_arr[:,0])
        self.Tf = self.boundary.params["Tf"]
        self.epsilon = self.boundary.params["epsilon"]
        self.Tf2 = self.Tf*(self.epsilon**2)
        
        self.A = self.boundary.afun()
        self.B = self.boundary.bfun()
        self.C = 2*(self.A - self.B) # (self.boundary.afun() - self.boundary.bfun())

        self.qaxis = np.unique(overdamped_arr[:,1])
        #self.reference.get_quantity(0,"value",arr=self.overdamped_arr)
        self.xnum = len(self.qaxis) #int(self.overdamped_arr.shape[0]/len(self.t2vec))

        self.filter_delta = 40 
        
    def apply_filter(self,y,mode="nearest"):
       
        return generic_filter(y,sc.median,size=self.filter_delta,mode=mode)
 
    def get_rhomask(self,t2):
        
        #return index where probability mass is sufficiently large
        return np.where(self.f00(t2)>0)

    def get_t0(self,t2):

        return round(t2/(self.epsilon**2),6)

    def v00(self,t2):
       
        return np.squeeze(self.overdamped_arr[np.where(self.overdamped_arr[:,0]==t2),-1]) #v00 

    def f00(self,t2):

        #_,f00 = self.reference.get_quantity(t2,"marginal",arr=self.overdamped_arr)
        return np.squeeze(self.overdamped_arr[np.where(self.overdamped_arr[:,0]==t2),2]) 

    def d_(self,y,x):

        return self.apply_filter(np.gradient(y,x))

    def save_perturbative(self,):
        """Compute perturbative drift and marginal distribution of the 
        position using analytic expressions.
        Results are added to a csv and saved
        """

        pertsol = np.c_[self.overdamped_arr, np.ones(self.xnum*len(self.t2vec)), np.ones(self.xnum*len(self.t2vec))] 
        
        for t in range(len(self.t2vec)):

            #perturbative drift 
            pertsol[t*self.xnum:(t+1)*self.xnum,-1] = self.DU(self.get_t0(self.t2vec[t]),self.t2vec[t])
            
            #perturbative marginal
            pertsol[t*self.xnum:(t+1)*self.xnum,-2] = self.marginal_perturbative(self.get_t0(self.t2vec[t]),self.t2vec[t])

        #save pertsol 
        np.savetxt(self.ref_path+"/perturbative_underdamped.csv",pertsol,fmt="%10.6f",header = "t,q,rho,V,f,DU",delimiter=',')
        
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
        np.savetxt(self.ref_path+"/cumulants.csv",pertcumulants,fmt="%10.6f",header = "t0,posmean,mommean,posvar,momvar,xcorr",delimiter=',')

class MarginalHalfBridgePerturbation(Perturbation):
    """Class to contain the functions for computing perturbative 
    corrections from overdamped data for the Marginal Half Bridge:
    when the terminal cost is specified by the KL divergence between the 
    realised final distribution and a reference one (the logarithm of their ratio)
    for the position marginal distribution
    """

    def __init__(self,boundary,overdamped_arr,ref_path):
        """
        initialises the class         
        args:
            boundary : class
                determines boundary conditions and parameters
            reference : class
                sets the input and output locations
        """

        Perturbation.__init__(self,boundary,overdamped_arr,ref_path)

        return

    def DU(self,t0,t2):
        """
        Compute the perturbative drift for the marginal half bridge
        args:
            t0 : float
                fast time
            t2 : float
                slow time
        return:
            du : ndarray
                analytic prediction of drift 
        """
        
        #get idx 
        idx = self.get_rhomask(t2)

        coeff = 1 - np.exp(t0-self.Tf)

        temp = np.zeros(self.xnum)
        temp = self.d_(self.apply_filter(self.v00(t2)),self.qaxis)

        return 2*coeff*temp

    def f02_term3(self,idx,v00_arr,f00_arr):

        #preallocate
        f02_temp = np.zeros(self.xnum)

        #compute gradient of V00
        f02_temp[idx] = f00_arr*self.d_(v00_arr,self.qaxis[idx]) 

        #take gradient
        f02_temp[idx] = self.d_(f02_temp[idx],self.qaxis[idx])

        return f02_temp 

    def f02_term2(self,idx,f00_arr):
        """Compute second deriv of f00_arr"""

        #preallocate
        f02_temp = np.zeros(self.xnum)

        #compute gradient 
        f02_temp[idx] = self.d_(f00_arr,self.qaxis[idx]) 

        #take gradient
        f02_temp[idx] = self.d_(f02_temp[idx],self.qaxis[idx])

        return f02_temp 


    def d_f00(self,t0,t2,idx,f00_arr):
        """Compute gradient of f00_arr"""

        #preallocate
        f02_temp = np.zeros(self.xnum)

        #compute gradient 
        f02_temp[idx] = self.d_(f00_arr,self.qaxis[idx]) 

        #take gradient
        #f02_temp[idx] = self.d_(f02_temp[idx],self.qaxis[idx])

        return f02_temp 

    def f02(self,t0,t2):
        
        coeff2 = (-t0*self.A + t0 + np.exp(-t0) - 1)
        coeff3 = -2*np.sinh(t0) + (1-np.exp(-self.Tf))*((t0/self.Tf)*(3-np.exp(-self.Tf))+2*(np.cosh(t0)-1))
        #-2*t0*self.B - 2*(np.sinh(t0)-t0-(np.cosh(t0)-1)*(1-np.exp(-self.Tf)))

        idx = self.get_rhomask(t2)

        v00_arr = self.apply_filter(self.v00(t2)[idx])
        f00_arr = self.apply_filter(self.f00(t2)[idx],mode="constant")
        
        term2 = self.f02_term2(idx,f00_arr)
        term3 = self.f02_term3(idx,v00_arr,f00_arr)
        
        return coeff2*term2 + coeff3*term3
    

    def f02_new(self,t0,t2):
        """Compute the f02 correction. 
        Uses the later version with a,b functions and only f00"""
        
        coeff2 = -(t0-(self.little_b(t0)/self.little_b(self.Tf)))
        coeff3 = (self.little_a(t0) - (self.little_a(self.Tf)*self.little_b(t0)/self.little_b(self.Tf)))
        #-2*np.sinh(t0) + (1-np.exp(-self.Tf))*((t0/self.Tf)*(3-np.exp(-self.Tf))+2*(np.cosh(t0)-1))

        idx = self.get_rhomask(t2)

        f00_arr = self.apply_filter(self.f00(t2)[idx],mode="constant")
        
        d_f00_2 = self.d_f00(idx,f00_arr)
        d_f00_3 = self.d_f00(idx,d_f00_2)
        
        return coeff2*d_f00_2 + coeff3*d_f00_3
        
    def marginal_perturbative(self,t0,t2):
        
        return self.f00(t2) + (self.epsilon**2)*self.f02_new(t0,t2)
        #self.f00(t2) + (self.epsilon**2)*self.f02(t0,t2)

    def little_a(self,t):
        return t - 1 + np.exp(-t)

    def little_b(self,t):
        return np.sinh(t)-t-(np.cosh(t)-1)*(1-np.exp(-self.Tf))

    def little_adot(self,t):
        return 1 - np.exp(-t)

    def little_bdot(self,t):
        return np.cosh(t)-1-(np.sinh(t))*(1-np.exp(-self.Tf))

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


class HalfBridgePerturbation(Perturbation):
    """Class to contain the functions for computing perturbative 
    corrections from overdamped data for the Half Bridge:
    when the terminal cost is specified by the KL divergence between the 
    realised final distribution and a reference one (the logarithm of their ratio)
    for the full phase space distribution
    """

    def __init__(self,boundary,overdamped_arr,ref_path):
        """
        initialises the class         
        args:
            boundary : class
                determines boundary conditions and parameters
            reference : class
                sets the input and output locations
        """

        Perturbation.__init__(self,boundary,overdamped_arr,ref_path)

        return

    def DU(self,t0,t2):
        """
        Compute the perturbative drift for the marginal half bridge
        args:
            t0 : float
                fast time
            t2 : float
                slow time
        return:
            du : ndarray
                analytic prediction of drift 
        """
        
        return 0


    def f02(self,t0,t2):
        
        return 0
        
    def marginal_perturbative(self,t0,t2):
        
        return self.f00(t2) + (self.epsilon**2)*self.f02(t0,t2)
        #self.f00(t2) + (self.epsilon**2)*self.f02(t0,t2)

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

if __name__=="__main__":
    from ..reference import RefSolution
    from ...sde.boundary import *
    
    boundary = UnderdampedLandauer()
    reference = RefSolution(boundary=boundary,folder_name="results_temp6")
    pertsol = MarginalHalfBridgePerturbation(boundary,reference)
    pertsol.save_perturbative()
    pertsol.save_perturbative_cumulants()