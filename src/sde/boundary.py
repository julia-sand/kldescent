"""
Includes classes defining the initial and reference distributions

    Boundaries: top level 

    OverdampedFullLandauer: Overdamped with Initial at 0 and reference as in final 2403.

    OverdampedLandauer: Overdamped with both Initial and reference as in final 2403.

    UnderdampedLandauer: Underdamped dynamics with Initial and reference are as in 2403.

    UnderdampedGaussian: Initial and reference are given by Gaussians, with moments can be parsed as arguments

    LinearOU: Linear Ornstein Uhlenbeck process in the overdamped dynamics with inital condition a delta peak defining a full bridge
"""

import os

import numpy as np 

from ..utils.parser import boundary_params, gaussian_params

class Boundaries():
    """
    Base class to store different boundary conditions

    Attributes
    ----------
    params : dict
        stores the boundary condition specific parameters
    Generator : class
        generator used for sampling from the distributions
    ref_path : str
        the file name of the default reference solution. 
    sigma0 : float
        sets the value of the volatility
    """

    def __init__(self,params=None,logger=None):
        """
        Initialises the class 

        Parameters
        ----------
        params : dict
            parameters of the boundary condition. If None, arguments are parsed from command line. If no arguments are parsed, defaults are used.
        logger : class
            logger object of the training run. If given, the parameters are logged through this object
        """

        self.sigma0 = 1 #np.sqrt(2)
        
        #get params 
        if params is not None:
            self.params = params
        else:
            self.params = boundary_params()
        
        self.Generator = np.random.default_rng(seed=42)

        if logger is not None:
            logger.log_dict(self.params) #add to log

        self.gauss_boundary = None #set final and initial means


        
class OverdampedFullLandauer(Boundaries):
    """
    Stores the boundary conditions for a bridge in the overdamped case, 
    where the initial state is a delta peak at 0 and the assigned final boundary is a double peak,
    See Arxiv refs. 2410. and 2403.  
    Attributes inherited from parent (Boundaries)

    Attributes
    ----------
    pf_norm : float
        normalisation constant for the final reference distribution

    Methods
    -------
    overwrite_ref_path()
        Overwrites the default reference solution path. 
    Ufinal()
        Returns the exponent of the assigned final distribution
    scorefinal()
        Returns the score of the assigned final distribution
    qfinal()
        Returns the unnormalised assigned final distribution
    q_marginal_final()
        Returns the normalised assigned final distribution
    generate_samples()
        Generates samples from the boundary distributions
    euler_maruyama_step()
        returns one step of the overdamped sde wrt problem description
    """
    
    def __init__(self,params=None,logger=None):
        Boundaries.__init__(self,params,logger)
        self.pf_norm = np.trapezoid(self.qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        #self.pi_norm = np.trapezoid(self.qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))       
        self.qmax = 6

    def overwrite_ref_path(self,new_path):
        
        self.ref_path = new_path + "/refsol.csv"

    def Ufinal(self,q):

        return ((q**2-1)**2) #*12

    def scorefinal(self,q):
        """compute negative log-gradient of the final distribution""" 
        return -(q**2-1)*q

    def qfinal(self,q):
        #unnormalised final reference marginal distribution
        return np.exp(-self.Ufinal(q)) 

    def q_marginal_final(self,q): 
        #normalised final reference distribution of the configuration
        return self.qfinal(q)/self.pf_norm

    #generate the initial data
    def generate_samples(self,batch_samples,mode):
        r"""sample points from the intial distribution for q 
        
        args:
            batch_samples : int
                number of samples to take
            mode : str
                initial or final. Return

        return 
            out : ndarray
                samples in the shape (batch_samples,1) from the final distribution if mode=="final", else zeros in shape(batch_samples,)
        """

        #from where to sample the q marginals
        if mode=="initial":
            out = np.zeros((batch_samples,1))

        elif mode=="final": 

            probs = self.q_marginal_final(np.linspace(-8,8,10000))/np.sum(self.q_marginal_final(np.linspace(-8,8,10000)))
            out = np.random.choice(np.linspace(-8,8,10000),
                                (batch_samples,1),
                                p=probs)
        
        return out
       
    def gfun(self,q):
        #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U
        return (self.Ufinal(q) - (q**2)/2)
 
    def f(self,q,t):

        return 0

    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):
        #takes one step forward in the euler maruyama discretisation
        return xn + (self.sigma0*du_bar)*dt + self.sigma0*dw

class OverdampedLandauer(OverdampedFullLandauer):

    def __init__(self,params=None,logger=None):
        OverdampedFullLandauer.__init__(self,params,logger)
        #self.pf_norm = np.trapezoid(self.qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        self.pi_norm = np.trapezoid(self.qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))       
        self.qmax = 6

    def Uinitial(self,q):
        return ((q-1)**4)#*12

    def qinitial(self,q):

        return np.exp(-self.Uinitial(q))

    def q_marginal_initial(self,q):

        return self.qinitial(q)/(self.pi_norm)

    def generate_samples(self,batch_samples,mode):
        r"""sample points from the intial distribution for q 
        
        args:
            batch_samples : int
                number of samples to take
            mode : str
                initial or final

        return 
            out : ndarray
                samples in the shape (batch_samples,) from the initial or final distribution 
        """

        #from where to sample the q marginals
        if mode=="initial":

            probs = self.q_marginal_initial(np.linspace(-8,8,10000))/np.sum(self.q_marginal_final(np.linspace(-8,8,10000)))

        elif mode=="final": 

            probs = self.q_marginal_final(np.linspace(-8,8,10000))/np.sum(self.q_marginal_final(np.linspace(-8,8,10000)))
        
        out = np.random.choice(np.linspace(-8,8,10000),
                            (batch_samples,1),
                            p=probs)
        
        return out

class LinearOU(Boundaries):
    def __init__(self,d=1):
        Boundaries.__init__(self)
        self.d = d #dimension #TODO!!! this needs to be added 
        self.gamma = 1
        self.A = 0.2
        self.Lambda = 1
        self.qmax = 6

    def q_marginal_initial(self,q):

        return np.exp(-(q**2))/np.sqrt(np.pi)

    def q_marginal_final(self,q): 
        
        return np.exp(-(q**2)/2)/np.sqrt(2*np.pi)

    def generate_samples(self,batch_samples,mode):
        r"""sample points from the intial distribution for q 
        
        args:
            batch_samples : int
                number of samples to take
            mode : str
                initial or final

        return 
            out : ndarray
                samples in the shape (batch_samples,1) from the initial or final distribution 
        """

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


    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):
        """
        One step forward in the euler maruyama discretisation of the overdamped dynamics

        args:
            xn : (n,2) arr
                current positions
            du_bar : (n,1) arr
                drift
            dt : float
                size of timestep
            dw : float
                innovation
            t : float
                current time
        returns
            x(n+1) : (n,2) arr
                positions after one timestep of the evolution
        """
        
        return xn + (self.b(xn,t) \
                     + self.sigma0*du_bar)*dt + self.sigma0*dw

class UnderdampedLandauer(OverdampedLandauer):
    def __init__(self,params=None,logger=None):
        OverdampedLandauer.__init__(self,params,logger)
        self.sigma0 = np.sqrt(2)
        self.qmax = 6

    def p_initial(self,q,p):

        return (self.q_marginal_initial(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)

    def p_final(self,q,p): 
        
        return (self.q_marginal_final(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)

    def log_pfinal(self,q,p):
        #unnormalised positive final log density 
        return self.Ufinal(q)+((p**2)/2)

    #generate the initial data
    def generate_samples_q(self,batch_samples,mode):
        r"""Sample points from the assigned configuration (position) distributions
        
        args:
            batch_samples : int
                number of samples to take of position 
            mode : str
                either "initial" or "final" whether to take from the initial or final distributions
        returns:
            xout : ndarray 
                samples from "mode" distribution with shape (batch_samples,)    
        """

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
        r"""Sample points from the assigned state distirbutions

        args:
            batch_samples : int
                number of samples to take of pairs (position,momentum) 
            mode : str
                either "initial" or "final" whether to take from the initial or final distributions
        returns:
            xout : ndarray 
                samples from "mode" distribution with shape(batch_samples,2)    
        """

        return np.column_stack((self.generate_samples_q(batch_samples,mode)
                                ,self.Generator.standard_normal(size=(batch_samples,1))))
    
    def f(self,x,t):

        return 0

    def b(self,x,t):

        return 0
    
    def euler_maruyama_step(self,xn,du_bar,dt,dw,t=0):

        #takes one step forward in the euler maruyama discretisation
        #go forward one step 
        xn[:,0,None] = xn[:,0,None] + self.params["epsilon"]*xn[:,1,None]*dt 
        xn[:,1,None] = xn[:,1,None] - (xn[:,1,None] - self.sigma0*du_bar)*dt \
                                    + self.sigma0*dw
        
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
       
class UnderdampedGaussian(UnderdampedLandauer):
    
    def __init__(self,params=None,
                        gauss_params=None,
                        logger=None):
        Boundaries.__init__(self,params,logger)
        self.sigma0 = np.sqrt(2)
        self.qmax = 25

        if gauss_params is not None: #read from file if available
            self.gauss_boundary = gauss_params
        else: #fetch from command line
            self.gauss_boundary = gaussian_params()

        if logger is not None:
            logger.log_dict(self.gauss_boundary) #add to log
        
        self.pi_norm = (np.sqrt(2*np.pi)*(self.gauss_boundary["sigmainitial"]))
        self.pf_norm = (np.sqrt(2*np.pi)*(self.gauss_boundary["sigmafinal"]))
        

    def overwrite_ref_path(self,new_path):
        
        self.ref_path = new_path + "/refsol.csv"
        self.cumulants_path = new_path + "/refcumulants.csv"

    def Uinitial(self,q):

        return (q-self.gauss_boundary["muinitial"])**2/(2*(self.gauss_boundary["sigmainitial"]**2))

    def Ufinal(self,q):

        return (q-self.gauss_boundary["mufinal"])**2/(2*(self.gauss_boundary["sigmafinal"]**2))


    def scorefinal(self,q):

        return -(q-self.gauss_boundary["muinitial"])/((self.gauss_boundary["sigmainitial"]**2))


    def q_marginal_initial(self,q):
        """
        args:
            q: position coordinate
        returns:
            pdf: normalised initial position marginal evaluated at q
        """

        return self.qinitial(q)/self.pi_norm

    def final_full_distribution_cov(self,):
        return np.array([[self.gauss_boundary["sigmafinal"]**2,0],[0,1]])

    def final_full_distribution_covdet(self,):
        return self.gauss_boundary["sigmafinal"]**2

    def final_full_distribution_inverse_cov(self,):
        return np.array([[(1/self.gauss_boundary["sigmafinal"]**2),0],[0,1]])

    def q_marginal_final(self,q): 
        
        return self.qfinal(q)/self.pf_norm

    def generate_samples_q(self,batch_samples,mode):
        r"""Sample points from the intial or final distributions for q (position)
        
        args:
            batch_samples : int
                number of samples to take
            mode : str
                initial or final

        return 
            out : ndarray
                samples in the shape (batch_samples,) from the initial or final distribution 
        """

        #from where to sample the q marginals
        if mode=="initial":
            out = self.Generator.normal(loc=self.gauss_boundary["muinitial"],scale=self.gauss_boundary["sigmainitial"],size=(batch_samples,1))     
        elif mode=="final": 
            out = self.Generator.normal(loc=self.gauss_boundary["mufinal"],scale=self.gauss_boundary["sigmafinal"],size=(batch_samples,1))     
            
        return out                           


if __name__=="__main__":

    boundary = UnderdampedGaussian()
    import matplotlib.pyplot as plt

    plt.plot(np.linspace(0,20),boundary.bfun(np.linspace(0,20))/boundary.afun(np.linspace(0,20)))
    plt.savefig("BA.png")
    plt.close()
