import numpy as np

class MarginalHalfBridgePerturbation():
    """Class to contain the functions for computing perturbative 
    corrections from overdamped data
    args:
        boundary : class
            determines boundary conditions and parameters
    """

    def __init__(self,boundary,reference):

        self.boundary = boundary
        self.reference = reference

        #load the csv
        self.overdamped_arr = self.reference.get_ref_data(data="overdamped")

        #set up expansion parameters
        self.t2vec = np.unique(self.overdamped_arr[:,0])
        self.xnum = int(self.overdamped_arr.shape[0]/len(self.t2vec))
        self.Tf = self.boundary.params["Tf"]
        self.epsilon = self.boundary.params["epsilon"]
        self.Tf2 = self.Tf*(self.epsilon**2)

    def v00(self,t2):

        x,v00 = self.reference.get_quantity(t2,"value",arr=self.overdamped_arr)
        return v00 

    def f00(self,t2):

        x,f00 = self.reference.get_quantity(t2,"marginal",arr=self.overdamped_arr)
        return f00

    def dv00(self,t2):

        x,v00 = self.reference.get_quantity(t2,"value",arr=self.overdamped_arr)
        return np.gradient(v00,x) 
    
    def ddv00(self,t2):

        x,v00 = self.reference.get_quantity(t2,"value",arr=self.overdamped_arr)
        return np.gradient(np.gradient(v00,x) ,x) 
    
    def df00(self,t2):

        x,f00 = self.reference.get_quantity(t2,"marginal",arr=self.overdamped_arr)
        return np.gradient(f00,x) 

    def ddf00(self,t2):

        x,f00 = self.reference.get_quantity(t2,"marginal",arr=self.overdamped_arr)
        return np.gradient(np.gradient(f00,x),x) 

    def v11(self,t0,t2):
        
        coeff = 1- np.exp(t0-self.Tf)

        return coeff*self.dv00(t2)

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
        
        term1 = -t0*self.df00(t2)
        term2 = self.ddf00(t2)*(t0+np.exp(-t0)-1)
        term3 = -2*(np.sinh(t0)-t0-(np.cosh(t0)-1)*(1-np.exp(-self.Tf)))*(self.df00(t2)*self.dv00(t2) + self.f00(t2)*self.ddv00(t2))    
                
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
            pertsol[t*self.xnum:(t+1)*self.xnum,-1] = self.DU(self.t2vec[t]/(self.epsilon**2),self.t2vec[t])
            
            #perturbative marginal
            pertsol[t*self.xnum:(t+1)*self.xnum,-2] = self.marginal_perturbative(self.t2vec[t]/(self.epsilon**2),self.t2vec[t])

        #save pertsol 
        np.savetxt(self.reference.get_ref_path()+"/perturbative_underdamped.csv",pertsol,fmt="%10.6f",header = "t,q,rho,V,f,DU",delimiter=',')
        #self.reference.add_perturbative_underdamped_data(self.reference.get_ref_path()+"/perturbative_underdamped.csv")

    def save_perturbative_cumulants(self,):
        """
        computes estiamtes of the perturbative cumulants and saves as a csv
        """

        #preallocate
        pertcumulants = np.empty((len(t2vec,6)))
        
        for t in range(len(self.t2vec)):
            
            pertcumulants[t,0] = self.t2vec[t]
            pertcumulants[t,1] = self.posmean(self.t2vec[t])
            pertcumulants[t,2] = self.mommean(self.t2vec[t])
            pertcumulants[t,3] = self.posvar(self.t2vec[t])
            pertcumulants[t,4] = self.momvar(self.t2vec[t])
            pertcumulants[t,5] = self.xcorr(self.t2vec[t])

        #save pertsol 
        np.savetxt(self.reference.get_ref_path()+"/cumulants.csv",pertcumulants,fmt="%10.6f",header = "t,posmean,mommean,posvar,momvar,xcorr",delimiter=',')

if __name__=="__main__":
    #from ..reference import RefSolution
    #from ...sde.boundary import *
    
    boundary = UnderdampedLandauer()
    reference = RefSolution(boundary=boundary)
    pertsol = MarginalHalfBridgePerturbation(boundary,reference)
    pertsol.save_perturbative()
    pertsol.save_perturbative_cumulants()