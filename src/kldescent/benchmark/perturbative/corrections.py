import numpy as np

from kldescent.benchmark.utilfunctions import *
from kldescent.utils.misc import append_array_to_csv

class Perturbation():

    def __init__(self,boundary_params,
                        overdamped_arr,
                        ref_path,
                        ):
        """Initialise base perturbation class"""

        #self.boundary_params = boundary_params
        
        self.ref_path = ref_path

        #load the csv
        self.overdamped_arr = overdamped_arr

        #define parameters
        self.t2vec = np.unique(self.overdamped_arr[:,0])
        self.Tf = boundary_params["Tf"]
        self.epsilon = boundary_params["epsilon"]
        self.M = boundary_params["weight"] 
        self.Tf2 = self.Tf*(self.epsilon**2)
        
        self.qaxis = np.unique(overdamped_arr[:,1])
        self.dx = np.abs(self.qaxis[1]-self.qaxis[0])
        self.xnum = len(self.qaxis) #int(self.overdamped_arr.shape[0]/len(self.t2vec))

        #size of convolution filter is 0.5pc of the datavector. 
        self.filter_delta = int(np.floor(self.xnum*0.001))
        
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
        f = self.f00(t2)
        return np.where(zchop(f, tol=1e-3 * f.max()) > 0)

    def get_t0(self,t2):
        """find t0 from t2"""

        return np.round(t2/(self.epsilon**2),6)

    def sigma(self,t2):
        """fetch sigma at time t2"""

        return np.squeeze(self.overdamped_arr[np.where(self.overdamped_arr[:,0]==t2),-1]) #sigma 

    def f00(self,t2):
        """fetch rho at time t2"""

        return np.squeeze(self.overdamped_arr[np.where(self.overdamped_arr[:,0]==t2),2]) 

    def d_(self,y,x):
        """compute the first derivative dy/dx 
        y and x must have same length"""
        
        return np.gradient(y,x)

    def dd_(self,y,x):
        """compute the second derivate d2y/dx2
        y and x must have same length"""
        
        return np.gradient(np.gradient(y,x),x)

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
        self.dmu_2_vec = np.gradient(self.mu_2_vec,self.t2vec)

    def get_sigma(self,t2,idx):
        sigma_raw = self.sigma(t2)[idx]
        sigma = savgol_filter_(sigma_raw,dx=self.dx,deriv=0)
        dsigma = savgol_filter_(sigma_raw,dx=self.dx,deriv=1)
        ddsigma = savgol_filter_(sigma_raw,dx=self.dx,deriv=2)
        #sigma = apply_filter(self.sigma(t2),filter_delta=self.filter_delta)[idx]
        #dsigma = self.d_(sigma,self.qaxis[idx])
        #ddsigma = self.d_(dsigma,self.qaxis[idx])
        return sigma, dsigma, ddsigma

    def get_rho(self,t2,idx):
        sigma_raw = self.f00(t2)[idx]
        f00 = savgol_filter_(sigma_raw,dx=self.dx,deriv=0)
        df00 = savgol_filter_(sigma_raw,dx=self.dx,deriv=1)
        ddf00 = savgol_filter_(sigma_raw,dx=self.dx,deriv=2)
        
        #f00 = apply_filter(self.f00(t2),filter_delta=self.filter_delta)[idx]
        #df00 = self.d_(f00,self.qaxis[idx])
        #ddf00 = self.d_(df00,self.qaxis[idx])
        return f00, df00, ddf00

    def kappa(self,f00,dsigma,idx):
        return np.trapezoid(f00*dsigma,self.qaxis[idx])/(self.A-self.B)
        
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
                
            idx = self.get_rhomask(t2)
            
            f00,df00,ddf00 = self.get_rho(t2,idx)
            
            sigma,dsigma,ddsigma = self.get_sigma(t2,idx)
            
            kappa = self.kappa(f00,dsigma,idx)
            
            arr_temp = np.c_[t0*np.ones_like(self.qaxis[idx]),
                                self.qaxis[idx], 
                                self.marginal_perturbative(t0,f00,df00,dsigma,ddf00,ddsigma,kappa),
                                self.DU(t0,f00,df00,dsigma,kappa)]

            append_array_to_csv(filename=filename_full,data=arr_temp[arr_temp[:,2] != 0])
     
    def save_perturbative_cumulants(self,):
        """Compute estimates of the perturbative cumulants and save to csv"""

        #preallocate array
        pertcumulants = np.empty((len(self.t2vec),6))
        
        #compute mu vectors in time. 
        self.compute_mu_vecs()

        for t2_idx in range(len(self.t2vec)):
                  
            t0 = self.get_t0(self.t2vec[t2_idx])
            t2 = self.t2vec[t2_idx]
            idx = self.get_rhomask(t2)
            
            f00, df00, ddf00 = self.get_rho(t2,idx)
            sigma,dsigma,ddsigma = self.get_sigma(t2,idx)

            kappa = self.kappa(f00,dsigma,idx)

            pertcumulants[t2_idx,0] = self.get_t0(t2)
            pertcumulants[t2_idx,1] = self.posmean(t0,t2,t2_idx,kappa)
            pertcumulants[t2_idx,2] = self.mommean(t0,kappa)
            pertcumulants[t2_idx,3] = self.posvar(t0,t2_idx)
            pertcumulants[t2_idx,4] = self.momvar(t0,t2,t2_idx,f00,df00,ddf00,dsigma,ddsigma,kappa,idx)
            pertcumulants[t2_idx,5] = self.xcorr(t0,t2,t2_idx,kappa,f00,dsigma,idx)#,self.t2vec[t],pertcumulants[t,1],pertcumulants[t,2])

        #save pertsol 
        np.savetxt(self.ref_path+"/perturbative_cumulants.csv",pertcumulants,fmt="%10.6f",header = "t0,posmean,mommean,posvar,momvar,xcorr",delimiter=',')
   
class MarginalHalfBridgePerturbation(Perturbation):
    """Class to contain the functions for computing perturbative 
    corrections from overdamped data for the Marginal Half Bridge:
    when the terminal cost is specified by the KL divergence between the 
    realised final distribution and a reference one (the logarithm of their ratio)
    for the position marginal distribution
    """

    def __init__(self,boundary_params,overdamped_arr,ref_path):
        """
        Initialise the class.

        Parameters
        ----------
        boundary_params : dict
            Determines boundary conditions and parameters.
        """

        Perturbation.__init__(self,boundary_params,overdamped_arr,ref_path)

        self.B = bfun("marginal half",self.Tf)
        self.A = afun("marginal half",self.Tf)
    
        return

    def DU(self,t0,f00,df00,dsigma,kappa):
        """
        Compute the perturbative drift for the marginal half bridge.

        Parameters
        ----------
        t0 : float
            Fast time.
        t2 : float
            Slow time.

        Returns
        -------
        du : np.ndarray
            Analytic prediction of drift.
        """
        
        raise NotImplementedError
    
    def f02_term2(self,idx,f00_arr):
        """Compute second deriv of f00_arr"""

        #preallocate
        f02_temp = np.zeros(self.xnum)

        #compute gradient 
        f02_temp[idx] = self.d_(f00_arr,self.qaxis[idx]) 

        #take gradient
        f02_temp[idx] = self.d_(f02_temp[idx],self.qaxis[idx])

        return f02_temp 

            
    def f02(self,t0,t2):#f00,df00,dsigma,ddf00,ddsigma,kappa):
        """Compute the f02 correction. 
        Uses the later version with a,b functions and only f00"""

        coeff2 = -(t0-self.Tf*(self.little_b(t0)/self.little_b(self.Tf)))
        coeff3 = (self.little_a(t0) - (self.little_a(self.Tf)*self.little_b(t0)/self.little_b(self.Tf)))

        idx = self.get_rhomask(t2)

        f00_arr = apply_filter(self.f00(t2)[idx],filter_delta=self.filter_delta)
        
        d_f00_2 = self.d_f00(idx,f00_arr)
        d_f00_3 = self.d_f00(idx,d_f00_2[idx])
        
        return coeff2*d_f00_2 + coeff3*d_f00_3
        
    def marginal_perturbative(self,t0,t2):#f00,df00,dsigma,ddf00,ddsigma,kappa):
        
        return self.f00(t2) + (self.epsilon**2)*self.f02(t0,t2)#f00,df00,dsigma,ddf00,ddsigma,kappa)

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


class HalfBridgePerturbation(Perturbation):
    """Class to contain the functions for computing perturbative 
    corrections from overdamped data for the Half Bridge:
    when the terminal cost is specified by the KL divergence between the 
    realised final distribution and a reference one (the logarithm of their ratio)
    for the full phase space distribution

    This reconstruction is for U_star=0 and keeps the leading 1/M
    approximation for X_M. The cell problem and public method signatures
    are unchanged. Cumulants use the same retained density coefficients.
    """

    def __init__(self,boundary_params,overdamped_arr,ref_path):
        """
        Initialise the class.

        Parameters
        ----------
        boundary_params : dict
            Determines boundary conditions and parameters.
        overdamped_arr : np.ndarray
            Array with the results of the cell problem integration.
        ref_path : str
            Path to access and save results.
        """

        Perturbation.__init__(self,boundary_params,overdamped_arr,ref_path)

        self.B = bfun("half",self.Tf)
        self.A = afun("half",self.Tf)
        self.compute_mu_vecs()
        return 

    def kappa(self,f00,dsigma,idx):
        """Normalized spatial expectation on the retained quadrature grid."""
        q = self.qaxis[idx]
        mass = np.trapezoid(f00,q)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("The retained density must have positive finite mass.")
        return np.trapezoid(f00*dsigma,q)/((self.A-self.B)*mass)

    def f02(self,t0,f00,df00,dsigma,ddf00,ddsigma,kappa):
        
        term1 = (ddf00*(np.cosh(t0)-1))/(2*self.M*np.sinh(self.Tf))
        coeff1_term2 = self.G110(t0)-(t0/self.Tf)*self.G110(self.Tf)
        coeff2_term2 = self.coeff_G111_G112()*(self.G114(t0)-(t0/self.Tf)*self.G114(self.Tf)+self.dt_G12(self.Tf)*(np.cosh(t0)-1)/(2*self.M*np.sinh(self.Tf)))
        term2 = kappa*df00/self.A
        coeff1_term34 = coeff1_term2#self.G110(t0)-(t0/self.Tf)*self.G110(self.Tf)
        coeff2_term34 = -coeff2_term2#self.coeff_G111_G112()*(self.G114(t0)-(t0/self.Tf)*self.G114(self.Tf)+self.dt_G12(self.Tf)*(np.cosh(t0)-1)/(2*self.M*np.sinh(self.Tf)))
        term3 = (df00*dsigma + f00*ddsigma)/self.A
        term4 = ddf00/np.sqrt(self.A)
        # Restore the d_t0 G11(Tf) contribution to -c(t0)*partial_q X_M.
        term5 = (self.dt_G11(self.Tf)*(np.cosh(t0)-1)/(2*self.M*np.sinh(self.Tf))
                 * (term3+term4+self.B*term2))
        return term1 + (self.B*coeff1_term2 + (self.A-self.B)*coeff2_term2)*term2 + (coeff1_term34+coeff2_term34)*(term3+term4) + term5

    def marginal_perturbative(self,t0,f00,df00,dsigma,ddf00,ddsigma,kappa):
        
        return f00 + (self.epsilon**2)*self.f02(t0,f00,df00,dsigma,ddf00,ddsigma,kappa)

    def DU(self,t0,f00,df00,dsigma,kappa):
        r"""
        Compute the perturbative drift for the half bridge
        args:
            t0 : float
                fast time
            f00 : ndarray
                distribution from cell problem
            df00 : ndarray
                gradient of distribution from cell problem
            dsigma : ndarray
                gradient of sigma from cell problem
            kappa : float
                value of varkappa, given by 
                \int_R f00 \partial \sigma 
        return:
            du : ndarray
                perturbative prediction of drift 
        """

        # XM is a density current; the potential gradient contains XM/f00.
        return self.dU_tilde(t0,f00,df00,dsigma,kappa) - self.coeff_XM(t0)*self.XM(t0,f00,df00,dsigma,kappa)/f00

    def dU_tilde(self,t0,f00,df00,dsigma,kappa):
        
        term1 = np.nan_to_num(df00/f00,nan=0,posinf=0,neginf=0)
        term2 = self.coeff_G111_G112()*kappa*(self.A-self.B)*(self.G12(t0)+self.dt_G12(t0))/self.A
        term3 = kappa*self.B*(self.G11(t0)+self.dt_G11(t0))/self.A
        coeff_t4 = self.G11(t0) + self.dt_G11(t0) - (self.G12(t0)+self.dt_G12(t0))*self.coeff_G111_G112()
        term4 = term1/np.sqrt(self.A) + dsigma/self.A

        return -term1-term2-term3-coeff_t4*term4

    def coeff_G111_G112(self,):
        return 3/(np.exp(-self.Tf)*(2+np.cosh(self.Tf)))

    def f11(self,f11_term1,f11_term2,t0):
        return self.G11(t0)*f11_term1 + self.G12(t0)*f11_term2
     
    def df11(self,f11_term1,f11_term2,t0):
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
        return ((2*np.exp(t0)-1)*(np.exp(t0-2*self.Tf))-(2*np.cosh(t0))/(1+np.exp(self.Tf)))/3

    def coeff_XM(self,t0):
        return (np.cosh(t0)+np.sinh(t0))/np.sinh(self.Tf)

    def XM(self,t0,f00,df00,dsigma,kappa):
        term1 = -df00/(2*self.M)
        term2_coeff = self.coeff_G111_G112()*self.dt_G12(self.Tf)/(2*self.M*self.A)
        term2_1 = kappa*(self.A-self.B)*f00
        term2_2 = f00*dsigma
        term2_3 = np.sqrt(self.A)*df00
        # Complete derivative of the tilde current at Tf, at leading 1/M.
        term3 = -self.dt_G11(self.Tf)/(2*self.M)*(df00/np.sqrt(self.A)
                    + f00*(dsigma+self.B*kappa)/self.A)
        return term1 - term2_coeff*(term2_1 - term2_2 - term2_3) + term3

    def G110(self,t0):
        return t0 - np.sinh(t0) +(np.cosh(t0)-1)*np.tanh(self.Tf/2)

    def G114(self,t0):
        return 2*((np.exp(-self.Tf + t0)*(1-np.exp(-t0))**2)/12)*((2/(1+np.exp(-self.Tf)))-np.exp(-self.Tf+t0))

    def _cumulants(self,t0,f00,df00,ddf00,dsigma,ddsigma,kappa,idx):
        """Moments of phi*(f00 + epsilon*p*f11 + epsilon**2*(f02+H2*f22)).

        The normalization is expanded to the same order on the retained
        spatial grid. This also keeps density and cumulant exports consistent.
        """
        q = self.qaxis[idx]
        integrate = lambda y: np.trapezoid(y,q)
        mass = integrate(f00)
        if not np.isfinite(mass) or mass <= 0 or np.any(f00 <= 0):
            raise ValueError("Retained density must have positive finite mass and positive values.")
        posmean = integrate(q*f00)/mass
        posvar = integrate((q-posmean)**2*f00)/mass

        f11_term1 = df00/np.sqrt(self.A)+f00*(dsigma+self.B*kappa)/self.A
        df11_term1 = ddf00/np.sqrt(self.A)+(df00*(dsigma+self.B*kappa)+f00*ddsigma)/self.A
        f11_term2 = -self.coeff_G111_G112()*(f11_term1-kappa*f00)
        XM = self.XM(t0,f00,df00,dsigma,kappa)
        dXM = -(ddf00+self.dt_G11(self.Tf)*df11_term1
                -self.coeff_G111_G112()*self.dt_G12(self.Tf)
                  *(df11_term1-kappa*df00))/(2*self.M)
        f11 = self.f11(f11_term1,f11_term2,t0)+XM*np.sinh(t0)/np.sinh(self.Tf)
        f02 = self.f02(t0,f00,df00,dsigma,ddf00,ddsigma,kappa)

        # Closed convolutions of G11 and G12 with exp(-2*(t0-r)).
        weighted_G11 = ((np.exp(t0)-np.exp(-2*t0))/(3*(1+np.exp(self.Tf)))
                        +np.exp(self.Tf)*(np.exp(-t0)-np.exp(-2*t0))/(1+np.exp(self.Tf))
                        -(1-np.exp(-2*t0))/2)
        weighted_G12 = (np.exp(-2*self.Tf)*((np.exp(2*t0)-np.exp(-2*t0))/12
                            -(np.exp(t0)-np.exp(-2*t0))/9)
                        -(np.exp(t0)-3*np.exp(-t0)+2*np.exp(-2*t0))/(9*(1+np.exp(self.Tf))))
        weighted = weighted_G11-self.coeff_G111_G112()*weighted_G12
        if t0 == 0 or t0 == self.Tf:
            weighted = 0.
        endpoint_weight = (np.exp(t0)/6-np.exp(-t0)/2+np.exp(-2*t0)/3)/np.sinh(self.Tf)
        if t0 == 0:
            endpoint_weight = 0.
        f22 = f11**2/(2*f00)-f00*(
            weighted*(df11_term1-f11_term1*df00/f00)/f00
            +endpoint_weight*(dXM-XM*df00/f00)/f00)

        mass02 = integrate(f02)
        mommean1 = integrate(f11)/mass
        return np.array([
            posmean+self.epsilon**2*integrate((q-posmean)*f02)/mass,
            self.epsilon*mommean1,
            posvar+self.epsilon**2*(integrate((q-posmean)**2*f02)-posvar*mass02)/mass,
            1+self.epsilon**2*(2*integrate(f22)/mass-mommean1**2),
            self.epsilon*integrate((q-posmean)*f11)/mass,
        ])

    def _cumulants_at_time(self,t0,t2,kappa=None):
        idx = self.get_rhomask(t2)
        f00,df00,ddf00 = self.get_rho(t2,idx)
        _,dsigma,ddsigma = self.get_sigma(t2,idx)
        if kappa is None:
            kappa = self.kappa(f00,dsigma,idx)
        return self._cumulants(t0,f00,df00,ddf00,dsigma,ddsigma,kappa,idx)

    def momvar(self,t0,t2,t2_idx,f00,df00,ddf00,dsigma,ddsigma,kappa,idx):
        return self._cumulants(t0,f00,df00,ddf00,dsigma,ddsigma,kappa,idx)[3]

    def integrand_3(self,s,t0):
        return np.exp(-2 * (t0 - s)) * (self.G11(s) - self.coeff_G111_G112()*self.G12(s))

    def posvar(self,t0,t2_idx):
        return self._cumulants_at_time(t0,self.t2vec[t2_idx])[2]

    def mommean(self,t0,kappa):
        t2 = self.t2vec[np.argmin(np.abs(self.t2vec-t0*self.epsilon**2))]
        return self._cumulants_at_time(t0,t2,kappa)[1]

    def posmean(self,t0,t2,t2_idx,kappa):
        return self._cumulants_at_time(t0,t2,kappa)[0]

    def xcorr(self,t0,t2,t2_idx,kappa,f00,dsigma,idx):
        return self._cumulants_at_time(t0,t2,kappa)[4]

    def save_perturbative_cumulants(self):
        # Compute each spatial slice once, retaining the existing CSV format.
        pertcumulants = np.empty((len(self.t2vec),6))
        for t2_idx,t2 in enumerate(self.t2vec):
            t0 = self.get_t0(t2)
            pertcumulants[t2_idx,0] = t0
            pertcumulants[t2_idx,1:] = self._cumulants_at_time(t0,t2)
        np.savetxt(self.ref_path+"/perturbative_cumulants.csv",pertcumulants,
                   fmt="%10.6f",header="t0,posmean,mommean,posvar,momvar,xcorr",delimiter=',')


class FullBridgePerturbation(MarginalHalfBridgePerturbation):
    """Functions for computing corrections from the cell problem for
    a full bridge transition between fixed endpoint data.
    Part of the corrections in the half bridge perturbation theory

    """

    def __init__(self, boundary_params, overdamped_arr, ref_path):
        super().__init__(boundary_params, overdamped_arr, ref_path)

        self.B = bfun("full", self.Tf)
        self.A = afun("full", self.Tf)
        self.A_minus_B = self.A - self.B
        self.alpha = np.sqrt(self.A)

    # --- basis functions 

    def a_minus_b2(self, t0):
        return 1 - (np.cosh(t0) + np.cosh(t0 - self.Tf)) / (1 + np.cosh(self.Tf))

    def b_fn(self, t0):
        denom = np.cosh(self.Tf)*np.sinh(self.Tf) - (2*np.sinh(self.Tf)*np.cosh(self.Tf) + 2*np.sinh(self.Tf))
        num = np.sinh(self.Tf - t0)*np.exp(-self.Tf) + np.sinh(t0)*np.exp(self.Tf) - np.sinh(self.Tf)*np.exp(2*t0 - self.Tf)
        return num / denom

    def a_fn(self, t0):
        return self.a_minus_b2(t0) + self.b_fn(t0)

    def b_dot(self, t0):
        denom = np.cosh(self.Tf)*np.sinh(self.Tf) - (2*np.sinh(self.Tf)*np.cosh(self.Tf) + 2*np.sinh(self.Tf))
        num = (-np.cosh(self.Tf - t0)*np.exp(-self.Tf)
               + np.cosh(t0)*np.exp(self.Tf)
               - 2*np.sinh(self.Tf)*np.exp(2*t0 - self.Tf))
        return num / denom

    def a_dot(self, t0):
        term1 = -(np.sinh(t0) + np.sinh(t0 - self.Tf)) / (1 + np.cosh(self.Tf))
        return self.b_dot(t0) + term1

    # --- helper integrals ---

    def dsigma_alpha_rho(self, f00, df00, dsigma):
        return dsigma + self.alpha * np.nan_to_num(df00 / f00, nan=0, posinf=0, neginf=0)

    def rho_ddsigma_alpha_rho_fn(self, f00, df00, ddf00, ddsigma, idx):
        dlogrho = np.nan_to_num(df00 / f00, nan=0, posinf=0, neginf=0)
        ddlogrho = np.nan_to_num(ddf00/f00, nan=0, posinf=0, neginf=0) - dlogrho**2
        #ddf00 / f00 - ((df00/f00)**2)#np.gradient(dlogrho, self.qaxis[idx])
        return f00 * (self.alpha * ddlogrho + ddsigma)

    def script_k_full(self, f00, df00, dsigma, idx, t2_idx, kappa):
        rho_dsig_alpha = f00 * dsigma + self.alpha * df00
        return -np.trapezoid(self.qaxis[idx] * rho_dsig_alpha, self.qaxis[idx]) + self.A_minus_B * kappa * self.mu_1_vec[t2_idx]

    # --- perturbative distribution and drift ---

    def f02(self, t0, t2_idx, f00, df00, ddf00, dsigma, ddsigma, idx, kappa, a_arr=None, b_arr=None):
        t0vec = self.t2vec / self.epsilon**2
        if a_arr is None:
            a_arr = np.array([self.a_fn(t) for t in t0vec])
        if b_arr is None:
            b_arr = np.array([self.b_fn(t) for t in t0vec])

        dsig_a = self.dsigma_alpha_rho(f00, df00, dsigma)
        rho_ddsig = self.rho_ddsigma_alpha_rho_fn(f00, df00, ddf00, ddsigma, idx)

        t2_term1 = df00 * dsig_a + rho_ddsig
        t2_termrho = -self.A_minus_B * kappa * df00

        coeff1_arr = -a_arr / self.A
        coeff2_arr = (self.B * a_arr - self.A * b_arr) / (self.A * self.A_minus_B)

        coeff1_int1 = (t0 / self.Tf) * np.trapezoid(coeff1_arr, t0vec)
        coeff2_int1 = (t0 / self.Tf) * np.trapezoid(coeff2_arr, t0vec)
        coeff1_int2 = np.trapezoid(coeff1_arr[:t2_idx+1], t0vec[:t2_idx+1])
        coeff2_int2 = np.trapezoid(coeff2_arr[:t2_idx+1], t0vec[:t2_idx+1])

        return (coeff1_int1 - coeff1_int2) * t2_term1 + (coeff2_int1 - coeff2_int2) * t2_termrho

    def marginal_perturbative(self, t0, t2_idx, f00, df00, dsigma, ddf00, ddsigma, idx, kappa, a_arr=None, b_arr=None):
        rvals = f00 + self.epsilon**2 * self.f02(t0, t2_idx, f00, df00, ddf00, dsigma, ddsigma, idx, kappa, a_arr, b_arr)
        return rvals / np.trapezoid(rvals, self.qaxis[idx])

    def DU(self, t0, f00, df00, dsigma, kappa):
        coeff1 = (self.a_dot(t0) + self.a_fn(t0)) / self.A
        dlogrho = np.nan_to_num(df00 / f00, nan=0, posinf=0, neginf=0)
        term1 = (self.alpha * coeff1 - 1) * dlogrho
        term2 = coeff1 * dsigma
        coeff3 = -kappa / self.A
        term3 = ((self.B * self.a_dot(t0) - self.A * self.b_dot(t0))
                 + (self.B * self.a_fn(t0) - self.A * self.b_fn(t0)))
        return np.nan_to_num(term1 + term2 - coeff3 * term3, nan=0, posinf=0, neginf=0)

    def save_perturbative(self):
        filename_full = make_csv_and_header(save_location=self.ref_path,
                                filename="perturbative_underdamped",
                                header="#t,q,f,DU")
        t0vec = self.t2vec / self.epsilon**2
        a_arr = np.array([self.a_fn(t) for t in t0vec])
        b_arr = np.array([self.b_fn(t) for t in t0vec])
        for t2_idx, t2 in enumerate(self.t2vec):
            t0 = self.get_t0(t2)
            idx = self.get_rhomask(t2)
            f00, df00, ddf00 = self.get_rho(t2, idx)
            sigma, dsigma, ddsigma = self.get_sigma(t2, idx)
            kappa = self.kappa(f00, dsigma, idx)
            arr_temp = np.c_[t0*np.ones_like(self.qaxis[idx]),
                                self.qaxis[idx],
                                self.marginal_perturbative(t0, t2_idx, f00, df00, dsigma, ddf00, ddsigma, idx, kappa, a_arr, b_arr),
                                self.DU(t0, f00, df00, dsigma, kappa)]
            append_array_to_csv(filename=filename_full, data=arr_temp[arr_temp[:,2] != 0])

    # --- cumulants ---

    def mommean(self, t0, t2,kappa):
        return -self.epsilon*kappa*(self.a_fn(t0)-self.b_fn(t0))

    def posmean(self, t0, t2, t2_idx, f00, dsigma, idx, kappa, a_arr=None, b_arr=None):
        t0vec = self.t2vec / self.epsilon**2
        if a_arr is None:
            a_arr = np.array([self.a_fn(t) for t in t0vec])
        if b_arr is None:
            b_arr = np.array([self.b_fn(t) for t in t0vec])
        ab_arr = a_arr - b_arr
        coeff2 = (self.epsilon**2) * kappa 
        int1 = (t0 / self.Tf) * np.trapezoid(ab_arr, t0vec)
        int2 = np.trapezoid(ab_arr[:t2_idx+1], t0vec[:t2_idx+1])
        return self.mu_1_vec[t2_idx] + coeff2 * (int1 - int2)

    def posvar(self, t0, t2, t2_idx, f00, df00, dsigma, idx, kappa, a_arr=None):
        t0vec = self.t2vec / self.epsilon**2
        if a_arr is None:
            a_arr = np.array([self.a_fn(t) for t in t0vec])
        sk = self.script_k_full(f00, df00, dsigma, idx, t2_idx, kappa)
        coeff3 = -2 * sk * (self.epsilon**2) / self.A
        int1 = (t0 / self.Tf) * np.trapezoid(a_arr, t0vec)
        int2 = np.trapezoid(a_arr[:t2_idx+1], t0vec[:t2_idx+1])
        return self.mu_2_vec[t2_idx] - self.mu_1_vec[t2_idx]**2 + coeff3 * (int1 - int2)

    def momvar(self, t0, t2, t2_idx, f00, df00, ddf00, dsigma, ddsigma, kappa, idx):
        t0vec = self.t2vec / self.epsilon**2
        term1 = 1 - (self.A_minus_B*kappa * self.epsilon * self.a_fn(t0) / self.A)**2
        rho_ddsig = self.rho_ddsigma_alpha_rho_fn(f00, df00, ddf00, ddsigma, idx)
        int1 = np.trapezoid(rho_ddsig, self.qaxis[idx]) / self.A
        aexp_arr = np.array([self.a_fn(t0vec[i]) * np.exp(-2*(t0 - t0vec[i])) for i in range(t2_idx+1)])
        int2 = int1 * np.trapezoid(aexp_arr, t0vec[:t2_idx+1])
        term2 = 2 * (self.epsilon**2) * int2
        dsig_a = self.dsigma_alpha_rho(f00, df00, dsigma)
        term3 = (self.epsilon * self.a_fn(t0) / self.A)**2 * np.trapezoid((dsig_a**2) * f00, self.qaxis[idx])
        return term1 + term2 + term3

    def xcorr(self, t0, t2, t2_idx, f00, df00, dsigma, idx, kappa):
        sk = self.script_k_full(f00, df00, dsigma, idx, t2_idx, kappa)
        return self.epsilon * self.a_fn(t0) * sk / self.A

    def save_perturbative_cumulants(self):
        pertcumulants = np.empty((len(self.t2vec), 6))
        self.compute_mu_vecs()
        t0vec = self.t2vec / self.epsilon**2
        a_arr = np.array([self.a_fn(t) for t in t0vec])
        b_arr = np.array([self.b_fn(t) for t in t0vec])
        for t2_idx in range(len(self.t2vec)):
            t0 = self.get_t0(self.t2vec[t2_idx])
            t2 = self.t2vec[t2_idx]
            idx = self.get_rhomask(t2)
            f00, df00, ddf00 = self.get_rho(t2, idx)
            sigma, dsigma, ddsigma = self.get_sigma(t2, idx)
            kappa = self.kappa(f00, dsigma, idx)
            pertcumulants[t2_idx, 0] = t0
            pertcumulants[t2_idx, 1] = self.posmean(t0, t2, t2_idx, f00, dsigma, idx, kappa, a_arr, b_arr)
            pertcumulants[t2_idx, 2] = self.mommean(t0, t2, kappa)
            pertcumulants[t2_idx, 3] = self.posvar(t0, t2, t2_idx, f00, df00, dsigma, idx, kappa, a_arr)
            pertcumulants[t2_idx, 4] = self.momvar(t0, t2, t2_idx, f00, df00, ddf00, dsigma, ddsigma, kappa, idx)
            pertcumulants[t2_idx, 5] = self.xcorr(t0, t2, t2_idx, f00, df00, dsigma, idx, kappa)
        np.savetxt(self.ref_path+"/perturbative_cumulants.csv", pertcumulants, fmt="%10.6f",
                   header="t0,posmean,mommean,posvar,momvar,xcorr", delimiter=',')

if __name__=="__main__":
    #from kldescent.sde.boundary import *
    from kldescent.utils.parser import fetch_results_folder_from_cmd,boundary_params

    boundary_params = boundary_params()

    folder_name = fetch_results_folder_from_cmd()

    #try:
    overdamped_arr = np.loadtxt(open(folder_name+"/overdamped.csv"),
                        delimiter=",",skiprows=1) 
        
    #compute perturbative
    pertsol = HalfBridgePerturbation(boundary_params,overdamped_arr,folder_name)
    #pertsol.save_perturbative()
    pertsol.save_perturbative_cumulants()
