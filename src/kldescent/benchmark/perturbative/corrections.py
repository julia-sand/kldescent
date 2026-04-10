import numpy as np

from kldescent.benchmark.utilfunctions import *

class Perturbation():

    def __init__(self,boundary,
                        overdamped_arr,
                        ref_path,
                        ):
        
        self.boundary = boundary
        
        self.ref_path = ref_path

        #load the csv
        self.overdamped_arr = overdamped_arr

        #define parameters
        self.t2vec = np.unique(self.overdamped_arr[:,0])
        self.Tf = self.boundary.params["Tf"]
        self.epsilon = self.boundary.params["epsilon"]
        self.Tf2 = self.Tf*(self.epsilon**2)
        
        self.qaxis = np.unique(overdamped_arr[:,1])
        self.xnum = len(self.qaxis) #int(self.overdamped_arr.shape[0]/len(self.t2vec))

        #size of convolution filter is 0.5pc of the datavector. 
        self.filter_delta = int(np.floor(self.xnum*0.05))

    def initialise_mu_vecs(self):

        #compute muvec for cumulants 
        self.mu_1_vec = np.zeros_like(self.t2vec)
        self.mu_2_vec = np.zeros_like(self.t2vec)
        self.dmu_1_vec = np.zeros_like(self.t2vec)
        self.dmu_2_vec = np.zeros_like(self.t2vec)
       
    def fill_zeros(self,y,idx):
        out = np.zeros(len(self.qaxis))
        out[idx] = y
        return out

    def get_rhomask(self,t2):
        """return index where rho is non zero at time t2"""
        
        return np.where(self.f00(t2)>0)

    def get_t0(self,t2):
        """find t0 from t2"""

        return round(t2/(self.epsilon**2),6)

    def dv00(self,t2):
        """fetch gradient of the value function at time t2"""

        return np.squeeze(self.overdamped_arr[np.where(self.overdamped_arr[:,0]==t2),-1]) #dv00 

    def f00(self,t2):
        """fetch rho at time t2"""

        return np.squeeze(self.overdamped_arr[np.where(self.overdamped_arr[:,0]==t2),2]) 

    def d_(self,y,x):
        """compute the gradient dy/dx and smooth 
        y and x must have same length"""
        
        return apply_filter(np.gradient(y,x),filter_delta=self.filter_delta)

    def mu_1_t2(self,t2):

        return np.trapezoid(self.qaxis*self.f00(t2),self.qaxis)

    def mu_2_t2(self,t2):

        return np.trapezoid((self.qaxis**2)*self.f00(t2),self.qaxis)

    def compute_mu_vecs(self):
        """compute mu expressions for cumulant calculations"""
        self.initialise_mu_vecs()

        for t2 in enumerate(self.t2vec):
            self.mu_1_vec[t2[0]] = self.mu_1_t2(t2[1])
            self.mu_2_vec[t2[0]] = self.mu_2_t2(t2[1])

        self.dmu_1_vec = np.gradient(self.mu_1_vec,self.t2vec)
        self.dmu_2_vec = np.gradient(self.mu_1_vec,self.t2vec)

    def save_perturbative(self,):
        """Compute perturbative drift and marginal distribution of the 
        position using analytic expressions.
        Results are added to a csv and saved
        """
        
        #make file with header 
        filename_full = make_csv_and_header(save_location=self.ref_path,
                                filename="perturbative_underdamped",
                                header="#t,q,f,DU")

        for t2 in self.t2vec:

            t0 = self.get_t0(t2)
            
            arr_temp = np.c_[t0*np.ones_like(self.qaxis),
                                self.qaxis, 
                                self.marginal_perturbative(t0,t2),
                                self.DU(t0,t2)]

            append_array_to_csv(filename=filename_full,data=arr_temp[arr_temp[:,2] != 0])
        
    def save_perturbative_cumulants(self,):
        """
        computes estiamtes of the perturbative cumulants and saves as a csv
        """

        #preallocate array
        pertcumulants = np.empty((len(self.t2vec),6))
        
        #compute mu vectors in time. 
        self.compute_mu_vecs()

        for t in range(len(self.t2vec)):
            
            pertcumulants[t,0] = self.get_t0(self.t2vec[t])
            pertcumulants[t,1] = self.posmean(self.get_t0(self.t2vec[t]),self.t2vec[t],t)
            pertcumulants[t,2] = self.mommean(self.get_t0(self.t2vec[t]),self.t2vec[t])
            pertcumulants[t,3] = self.posvar(self.get_t0(self.t2vec[t]),self.t2vec[t],t)
            pertcumulants[t,4] = self.momvar(self.get_t0(self.t2vec[t]),self.t2vec[t])
            pertcumulants[t,5] = self.xcorr(self.get_t0(self.t2vec[t]),self.t2vec[t],pertcumulants[t,1],pertcumulants[t,2])

        #save pertsol 
        np.savetxt(self.ref_path+"/perturbative_cumulants.csv",pertcumulants,fmt="%10.6f",header = "t0,posmean,mommean,posvar,momvar,xcorr",delimiter=',')

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
            benchmark : class
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

        #temp = np.zeros(self.xnum)
        #temp = apply_filter(self.dv00(t2))

        return 0#2*coeff*temp
    
    def f02_term2(self,idx,f00_arr):
        """Compute second deriv of f00_arr"""

        #preallocate
        f02_temp = np.zeros(self.xnum)

        #compute gradient 
        f02_temp[idx] = self.d_(f00_arr,self.qaxis[idx]) 

        #take gradient
        f02_temp[idx] = self.d_(f02_temp[idx],self.qaxis[idx])

        return f02_temp 

    def d_f00(self,idx,f00_arr):
        """Compute gradient of f00_arr"""

        #preallocate
        f02_temp = np.zeros(self.xnum)

        #compute gradient 
        f02_temp[idx] = apply_filter(self.d_(f00_arr,self.qaxis[idx]),filter_delta=self.filter_delta)

        return f02_temp
    
            
    def f02(self,t0,t2):
        """Compute the f02 correction. 
        Uses the later version with a,b functions and only f00"""

        coeff2 = -(t0-self.Tf*(self.little_b(t0)/self.little_b(self.Tf)))
        coeff3 = (self.little_a(t0) - (self.little_a(self.Tf)*self.little_b(t0)/self.little_b(self.Tf)))

        idx = self.get_rhomask(t2)

        f00_arr = apply_filter(self.f00(t2)[idx],filter_delta=self.filter_delta)
        
        d_f00_2 = self.d_f00(idx,f00_arr)
        d_f00_3 = self.d_f00(idx,d_f00_2[idx])
        
        return coeff2*d_f00_2 + coeff3*d_f00_3
        
    def marginal_perturbative(self,t0,t2):
        
        return self.f00(t2) + (self.epsilon**2)*self.f02(t0,t2)

    def little_a(self,t0):
        
        return t0 - 1 + np.exp(-t0)

    def little_b(self,t0):
        return np.sinh(t0)-t0-(np.cosh(t0)-1)*(1-np.exp(-self.Tf))

    def little_adot(self,t0):
        return 1 - np.exp(-t0)

    def little_bdot(self,t0):
        return np.cosh(t0)-1-(np.sinh(t0))*(1-np.exp(-self.Tf))

    def integral_log_rho(self,t2):
        
        idx = self.get_rhomask(t2)

        #get rho 
        f00_arr = apply_filter(self.f00(t2)[idx],filter_delta=self.filter_delta)
        
        #preallocate
        rho00 = np.zeros(self.xnum)
        rho00[idx] = -(self.d_(f00_arr,self.qaxis[idx])**2)/f00_arr
        
        #add ddrho
        rho00 += self.f02_term2(idx,f00_arr)
        
        return np.trapezoid(rho00,self.qaxis)

    def big_b(self,t0):
        
        coeff = self.little_a(self.Tf)/self.little_b(self.Tf)
        term1 = (np.sinh(t0)+self.little_bdot(t0))*(1+coeff)*np.exp(-t0)
        term2 = np.exp(-self.Tf)*(np.exp(-2*t0)-np.exp(t0))/3
        term3 = coeff*(np.exp(-self.Tf)*np.sinh(t0)+np.exp(-2*self.Tf)*(1-np.cosh(2*t0))/4)
        return coeff*(term1+term2-term3)

    def momvar(self,t0,t2):
        return 1 - 2*(self.epsilon**2)*self.big_b(t0)*self.integral_log_rho(t2)
        
    def posvar(self,t0,t2,t2_idx):
        term1 = (self.mu_2_vec[t2_idx]-(self.mu_1_vec[t2_idx]**2))
        term2 = 2*(self.little_a(t0)-(self.little_b(t0)/self.little_b(self.Tf))*self.little_a(self.Tf))
        term3 = (t0-(self.little_b(t0)/self.little_b(self.Tf))*self.Tf)*(self.dmu_2_vec[t2_idx]+2*self.mu_1_vec[t2_idx]*self.dmu_1_vec[t2_idx])
        return term1 + (self.epsilon**2)*(term2-term3)

    def mommean(self,t0,t2):
        """compute momentum mean"""
        return 0

    def posmean(self,t0,t2,t2_idx):
        """compute position mean"""
        return self.mu_1_vec[t2_idx] - self.dmu_1_vec[t2_idx]*(self.epsilon**2)*(t0 - self.Tf*(self.little_b(t0)/self.little_b(self.Tf)))

    def xcorr(self,t0,t2,posmean,mommean):
        """compute cross correllation"""
        return -self.epsilon*(self.little_bdot(t0)*self.little_a(self.Tf)/self.little_b(self.Tf)- self.little_adot(t0)) - (posmean*mommean)

class HalfBridgePerturbation(MarginalHalfBridgePerturbation):
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
            benchmark : class
                sets the input and output locations
        """

        Perturbation.__init__(self,boundary,overdamped_arr,ref_path)

        return 

    def DU(self,t0,t2):
        """
        Compute the perturbative drift for the half bridge
        args:
            t0 : float
                fast time
            t2 : float
                slow time
        return:
            du : ndarray
                analytic prediction of drift 
        """

        idx = self.get_rhomask(t2)
        
        f00 = self.fill_zeros(apply_filter(self.f00(t2)[idx],filter_delta=self.filter_delta),idx)
        df00 = self.d_f00(idx,f00[idx])
        dv00 = apply_filter(self.dv00(t2),filter_delta=self.filter_delta) 
        
        kappa = self.kappa(f00,dv00)
        f11_term1 = self.f11_term1(df00,f00,dv00)
        f11_term2 = self.f11_term2(df00,f00,dv00,kappa)
        
        f11 = self.f11(f11_term1,f11_term2,t0)
        dt_f11 = self.dt_f11(f11_term1,f11_term2,t0)
        
        du_out = np.nan_to_num((df00 + dt_f11 + f11)/f00,posinf=0,neginf=0)
        return -apply_filter(du_out,filter_delta=self.filter_delta)

    def coeff_G111_G112(self,):
        return 3/(np.exp(-self.Tf)*(2+np.cosh(self.Tf)))

    def f11(self,f11_term1,f11_term2,t0):
        return self.G11(t0)*f11_term1 + self.G12(t0)*f11_term2
     
    def dt_f11(self,f11_term1,f11_term2,t0):
        return self.dt_G11(t0)*f11_term1 + self.dt_G12(t0)*f11_term2
        
    def d_v20_f00(self,df00,f00,dv00,kappa):
        return -self.coeff_G111_G112()*(df00 + f00*(2*dv00 - kappa))/4

    def f11_term1(self,df00,f00,dv00):
        return df00+2*f00*dv00

    def f11_term2(self,df00,f00,dv00,kappa):
        return 4*self.d_v20_f00(df00,f00,dv00,kappa)

    def G11(self,t0):
        return 2*np.sinh(t0/2)*(np.sinh(t0/2)-np.cosh(t0/2)*np.tanh(self.Tf/2))

    def G12(self,t0):
        return -((1-np.exp(-t0))/3)*(((1+np.exp(t0))/(1+np.exp(self.Tf)))-np.exp(-2*(self.Tf-t0)))

    def dt_G11(self,t0):
        return np.sinh(t0)-np.cosh(t0)*np.tanh(self.Tf/2)

    def dt_G12(self,t0):
        return 2*(np.exp(t0)-1)*np.exp(t0-2*self.Tf)/3 - 2*np.cosh(t0)/(3*(1+np.exp(self.Tf)))

    def kappa(self,f00,dv00):
        return 2*np.trapezoid(f00*dv00,self.qaxis)
        
    def xcorr(self,t0,t2,posmean,mommean):
        return 0 -(posmean*mommean)

    def momvar(self,t0,t2):
        return 0

    def posvar(self,t0,t2,t2_idx):
        """Compute position variance
        args: 
        t0 : fast time
        t2 : slow time
        t2_idx : index of t2
        """

        return 0

    def mommean(self,t0,t2):
        return 0

    def posmean(self,t0,t2,t2_idx):
        return 0

if __name__=="__main__":
    from kldescent.benchmark.benchmarksol import PerturbativeBenchmarkSolution
    from kldescent.sde.boundary import *
    from kldescent.utils.parser import fetch_results_folder_from_cmd

    boundary = UnderdampedLandauer()

    folder_name = fetch_results_folder_from_cmd()

    #try:
    overdamped_arr = np.loadtxt(open(folder_name+"/overdamped.csv"),
                        delimiter=",",skiprows=1) 
        #compute overdamped solution
    #    compute_overdamped(boundary,
    #                        folder_name,
    #                        bridge_type="half")
    #    overdamped_arr = np.loadtxt(open(folder_name+"/overdamped.csv"),
    #                    delimiter=",",skiprows=1) 

    #compute perturbative
    pertsol = HalfBridgePerturbation(boundary,overdamped_arr,folder_name)
    pertsol.save_perturbative()
    pertsol.save_perturbative_cumulants()