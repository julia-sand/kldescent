import os

import numpy as np 

from ..utils.parser import boundary_params, gaussian_params

"""
Stores different boundary conditions for experiments

"""

class Boundaries():
    def __init__(self,params=None,logger=None):

        self.sigma0 = 1 #np.sqrt(2)
        
        #get params 
        if params is not None:
            self.params = params
        else:
            self.params = boundary_params()
            
        
        self.Generator = np.random.default_rng(seed=None)

        self.ref_path = self.get_ref_path()+"/refsol.csv"

        if logger is not None:
            logger.log_dict(self.params) #add to log
        
    def get_ref_path(self):
        #returns the folder where the reference solution is expected to be

        return os.path.dirname(__file__) + f"/../refsolution/{self.__class__.__name__}"
 

class HalfBridgeLandauer(Boundaries):
    def __init__(self,params=None,logger=None):
        Boundaries.__init__(self,params,logger)
        self.pf_norm = np.trapz(self.qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        self.pi_norm = np.trapz(self.qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))       


    def overwrite_ref_path(self,new_path):
        
        self.ref_path = new_path + "/refsol.csv"

    def Uinitial(self,q):

        return (q**4)/4

    def Ufinal(self,q):

        return ((q**2-1)**2)/4

    def scorefinal(self,q):

        return -(q**2-1)*q

    def qinitial(self,q):

        return np.exp(-self.Uinitial(q))

    def qfinal(self,q):

        return np.exp(-self.Ufinal(q)) 

    def q_marginal_initial(self,q):

        return self.qinitial(q)/(self.pi_norm)


    def q_marginal_final(self,q): 
        
        return self.qfinal(q)/self.pf_norm


    def p_initial(self,q,p):

        return (self.q_marginal_initial(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)


    def p_final(self,q,p): 
        
        return (self.q_marginal_final(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)

    #generate the initial data
    def generate_samples(self,batch_samples,mode):
        
        ##sample points from the intial distribution for q 
        #output as a tensor

        #from where to sample the q marginals
        if mode=="initial":
            out = np.zeros((batch_samples,1))
            #q_marginal = self.q_marginal_initial

        elif mode=="final": 
            q_marginal = self.q_marginal_final

            probs = q_marginal(np.linspace(-8,8,10000))/np.sum(q_marginal(np.linspace(-8,8,10000)))
            out = np.random.choice(np.linspace(-8,8,10000),
                                (batch_samples,1),
                                p=probs)
        
        return out
                            
       
    def get_reference_timegrid(self):
        arr = np.loadtxt(self.ref_path,
                        delimiter=",") 
        return np.unique(arr[:,0])

    def get_reference_data(self,time):

        #time is the time you want the data

        ##fetch the reference data for halfbridge
        arr = np.loadtxt(self.ref_path,
                        delimiter=",")  
       
        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,8))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        #RETURNS X, DU
        return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,-1])

    def du_ref(self,q,time):

        #returns the drift of the reference solution 
        qref,duref = self.get_reference_data(time)
        return np.interp(q,qref,duref,left=duref[0],right=duref[-1])

    def gfun(self,q):
        #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U

        return (self.Ufinal(q) - (q**2)/2)
 
    def f(self,q,t):

        return 0

    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):
        #takes one step forward in the euler maruyama discretisation
        
        return xn + (self.sigma0*du_bar)*dt + self.sigma0*dw


class LinearOU(Boundaries):
    def __init__(self,d=1):
        Boundaries.__init__(self)
        self.d = d #dimension #TODO!!! this needs to be added 
        self.gamma = 1
        self.A = 0.2
        self.Lambda = 1
        
    def q_marginal_initial(self,q):

        return np.exp(-(q**2))/np.sqrt(np.pi)

    def q_marginal_final(self,q): 
        
        return np.exp(-(q**2)/2)/np.sqrt(2*np.pi)
        #qfinal(q)/self.pf_norm

    #generate the initial data
    def generate_samples(self,batch_samples,mode):
        
        ##sample points from the intial distribution for q 
        #output as a tensor

        #from where to sample the q marginals
        if mode=="initial":
            out = self.Generator.normal(loc=0,scale=np.sqrt(0.5),size=(batch_samples,1))
        elif mode=="final": 
            out = np.zeros((batch_samples,1))
            #q_marginal = self.q_marginal_final
        
        return out

    def analytic_sol(self,time):
        return -self.sigma0*np.exp(self.A*(self.params["Tf"]-time))*self.gamma


    def get_reference_data(self,time):

        #time is the time you want the data
        
        ut = self.analytic_sol(time)

        #RETURNS X, DU
        return np.linspace(-4,4,num=50),ut*np.ones(50)


    def du_ref(self,q,time):

        ut = self.analytic_sol(time)
        
        return np.ones(q.shape)*ut


    def gfun(self,q):

        return self.gamma*q

    def f(self,q,t):

        return 0

    def b(self,q,t):

        return self.A*q

    def get_reference_timegrid(self,time=None):
         
        return None

    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):
        #takes one step forward in the euler maruyama discretisation
        
        return xn + (self.b(xn,t) \
                     + self.sigma0*du_bar)*dt + self.sigma0*dw

class UnderdampedHalfLandauer(HalfBridgeLandauer):
    def __init__(self,params=None,logger=None):
        HalfBridgeLandauer.__init__(self,params,logger)
        self.sigma0 = np.sqrt(2)

        self.pf_norm = np.trapz(self.qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        self.pi_norm = np.trapz(self.qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))

    def Uinitial(self,q):

        return ((q-1)**4)/4

    def Ufinal(self,q):

        return ((q**2-1)**2)/4

    def scorefinal(self,q):

        return -(q**2-1)*q

    def qinitial(self,q):

        return np.exp(-self.Uinitial(q))

    def qfinal(self,q):

        return np.exp(-self.Ufinal(q)) 

    def q_marginal_initial(self,q):

        return self.qinitial(q)/(self.pi_norm)


    def q_marginal_final(self,q): 
        
        return self.qfinal(q)/self.pf_norm


    def p_initial(self,q,p):

        return (self.q_marginal_initial(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)


    def p_final(self,q,p): 
        
        return (self.q_marginal_final(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)

    #generate the initial data
    def generate_samples_q(self,batch_samples,mode):
        
        ##sample points from the intial distribution for q 
        #output as a tensor

        #from where to sample the q marginals
        if mode=="initial":
            q_marginal = self.q_marginal_initial     
        elif mode=="final": 
            q_marginal = self.q_marginal_final

        probs = q_marginal(np.linspace(-8,8,10000))/np.sum(q_marginal(np.linspace(-8,8,10000)))
        out = np.random.choice(np.linspace(-8,8,10000),
                                (batch_samples,1),
                                p=probs)
            
        return out 
                            
       
    #generate the initial data
    def generate_samples(self,batch_samples,mode):
        
        ##sample points from the intial distribution for q 
        ##and from normal distribution for p and stack together 
        #output as a tensor

        #gives the initialisation of the underdamped
             
        return np.column_stack((self.generate_samples_q(batch_samples,mode)
                                ,self.Generator.standard_normal(size=(batch_samples,1))))
    
   
    def get_reference_timegrid(self,time=None):
        if time is None:
            time = self.params["Tf"]
        return np.linspace(0,time)

    def get_reference_data(self,time):

        #RETURNS X, DU for plotting
        return np.zeros(1000),np.zeros(1000)

    def du_ref(self,q,time):

        #returns the drift of the reference solution 
        #print("Reference solution not available.")
        return np.zeros_like(q)

    def gfun(self,x):

        #Terminal cost
        return (x[...,[1]]**2)/2 + self.Ufinal(x[...,[0]])

    def f(self,x,t):

        return 0

    def b(self,x,t):

        return 0
    
    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):

        #takes one step forward in the euler maruyama discretisation
        #go forward one step 
        xn[:,0,None] = xn[:,0,None] + self.params["epsilon"]*xn[:,1,None]*dt 
        xn[:,1,None] = xn[:,1,None] - (xn[:,1,None] - self.sigma0*du_bar)*dt \
                                    + self.sigma0*dw
        
        #xn[:,0,None] = xn[:,0,None] + (xn[:,1,None]/self.m)*dt 
        #xn[:,1,None] = xn[:,1,None] - (xn[:,1,None]/self.tau - du_bar)*dt \
        #                            + self.sigma0*dw

        return xn


    def overdamped_counterpart(self,xn,du_bar,dt,dw,t):

        #takes one step forward in the euler maruyama discretisation
        #go forward one step 
        xn = xn + du_bar*dt + self.sigma0*dw

        return xn

    def girsanov_scheme(self,xn,du_bar,dt,dw,t):

        #takes one step forward in the euler maruyama discretisation
        #go forward one step 
        xn[...,[0]] = xn[...,[0]] + self.params["epsilon"]*xn[...,[1]]*dt 
        xn[...,[1]] = xn[...,[1]] + self.sigma0*du_bar*dt + self.sigma0*dw

        return xn
        

class UnderdampedNoMomentum(UnderdampedHalfLandauer):
    def __init__(self,params=None,logger=None):
        super().__init__(self,params,logger)

    def gfun(self,x):
        #overwrite the terminal cost function 

        #Terminal cost
        return self.Ufinal(x[...,[0]]) #
        

    def get_reference_array(self):
        return np.loadtxt(self.ref_path,
                               delimiter=",")

    def get_reference_timegrid(self,plottime):
        
        #time is the time you want the data
        arr = self.get_reference_array()
        ttemp = np.round(np.unique(arr[:,0])/(self.params["epsilon"]**2),5) #rescale

        return ttemp[np.where(ttemp<plottime)]


    def get_reference_data(self,time):

        arr = self.get_reference_array()

        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time*self.params["epsilon"]**2,7))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        #coeff
        coeff = (1 - np.exp((time - self.params["Tf"])))

        #RETURNS X, DU for plotting
        return np.squeeze(arrtemp[:,1]),2*coeff*np.squeeze(arrtemp[:,-1])

    def get_reference_value(self,time):

        arr = self.get_reference_array()

        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time*self.params["epsilon"]**2,7))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x


        #RETURNS X, DU for plotting
        return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,-2])

    def du_ref(self,q,time):

      
        #returns the drift of the reference solution 
        qref,duref = self.get_reference_data(time)
        return np.interp(q,qref,duref,left=duref[0],right=duref[-1])[:, np.newaxis]


class UnderdampedQuarterGaussian(UnderdampedHalfLandauer):
    def __init__(self,params=None,gauss_params=None,logger=None):
        Boundaries.__init__(self,params,logger)
        self.sigma0 = np.sqrt(2)

        if gauss_params is not None: #read from file if available
            self.gauss_boundary = gauss_params
        else: #fetch from command line
            self.gauss_boundary = gaussian_params()

        if logger is not None:
            logger.log_dict(self.gauss_boundary) #add to log
        
    def Uinitial(self,q):

        return (q-self.gauss_boundary["muinitial"])**2/(2*(self.gauss_boundary["sigmainitial"]**2))

    def Ufinal(self,q):

        return (q-self.gauss_boundary["mufinal"])**2/(2*(self.gauss_boundary["sigmafinal"]**2))


    def scorefinal(self,q):

        return -(q-self.gauss_boundary["muinitial"])/((self.gauss_boundary["sigmainitial"]**2))


    def q_marginal_initial(self,q):

        return self.qinitial(q)/np.sqrt(2*(self.gauss_boundary["sigmainitial"]**2))


    def q_marginal_final(self,q): 
        
        return self.qfinal(q)/np.sqrt(2*(self.gauss_boundary["sigmafinal"]**2))


    #generate the initial data
    def generate_samples_q(self,batch_samples,mode):
        
        ##sample points from the intial distribution for q 
        #output as a tensor

        #from where to sample the q marginals
        if mode=="initial":
            out = self.Generator.normal(loc=self.gauss_boundary["muinitial"],scale=self.gauss_boundary["sigmainitial"],size=(batch_samples,1))     
        elif mode=="final": 
            out = self.Generator.normal(loc=self.gauss_boundary["mufinal"],scale=self.gauss_boundary["sigmafinal"],size=(batch_samples,1))     
            
        return out                           

    def get_ref_data(self):
        #returns the folder where the reference solution is expected to be
        arr = np.loadtxt(open(self.ref_path),
                        delimiter=",",skiprows=1) 
        return arr 

    def get_reference_timegrid(self,time=None):
        if time is None:
            time = self.params["Tf"] + 0.00001
        
        try: 
            arr = self.get_ref_data()
            return np.unique(arr[:,0])
        except FileNotFoundError:
            return np.linspace(0,time)

    def get_reference_data(self,time):
        arr = self.get_ref_data()
        
        return np.linspace(-4,4),-(np.squeeze(arr[np.where(arr[:,0]==time),-1]) + np.squeeze(arr[np.where(arr[:,0]==time),-2])*np.linspace(-4,4))

    def du_ref(self,q,time):
        #A FREE DIFFUSION
        # ie. returns zero drift of the right shape! 
        return np.zeros_like(q)

    def get_known_solution_cumulants(self,cumulant_type):
        arr = self.get_ref_data()#np.loadtxt(open(self.ref_path),
              #          delimiter=",",skiprows=1) 
        try:  
            if cumulant_type == "Position Variance": 
                return np.squeeze(arr[:,4])
            elif cumulant_type == "Momentum Variance":
                return np.squeeze(arr[:,6])
            elif cumulant_type == "Cross Correlation":
                return np.squeeze(arr[:,5])
            elif cumulant_type == "Momentum Mean":
                return np.squeeze(arr[:,8])
            elif cumulant_type == "Position Mean":
                return np.squeeze(arr[:,7])
            else:
                return np.zeros_like(np.squeeze(arr[:,0]))
        except IndexError:
            return    
 

class UnderdampedQuarterGaussianNoMomentum(UnderdampedQuarterGaussian):
    def __init__(self,params=None,gauss_params=None,logger=None):
        UnderdampedQuarterGaussian.__init__(self,params,gauss_params,logger)
        
    def gfun(self,x):
        #overwrite the terminal cost function 

        #Terminal cost
        return self.Ufinal(x[...,[0]])


class UnderdampedHalfGaussian(UnderdampedQuarterGaussian):
    def __init__(self,params=None,gauss_params=None,logger=None):
        UnderdampedQuarterGaussian.__init__(self,params,gauss_params,logger)
        self.mean0 = 0
        self.var0 = 1

    def set_ref_statistics(self,mean0,var0):
        """overwrites the mean and variance of the reference parameters"""

        self.mean0 = mean0
        self.var0 = var0

    def Utemp(self,q,mean,var):


        return (q-mean)**2/(var)

    def gfun(self,x):      
        #print(self.mean0)

        #Terminal cost
        return (self.params["weight"]/2)*(-self.Utemp(x[...,[0]],self.mean0,self.var0)+self.Utemp(x[...,[0]],self.gauss_boundary["mufinal"], self.gauss_boundary["sigmafinal"]**2))


if __name__=="__main__":

    boundary = UnderdampedQuarterGaussian()
    
