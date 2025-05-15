import numpy as np 
import os
from ..utils.parser import boundary_params
#the boundary conditions    


class HalfBridgeLandauer():
    def __init__(self):
        self.sigma0 = 1#np.sqrt(2)
        self.pf_norm = np.trapz(self.qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        self.pi_norm = np.trapz(self.qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))

        #get params 
        self.params = boundary_params()
        self.epsilon = self.params.epsilon 
        self.Tf = self.params.Tf
        self.g = self.params.g        
        
        self.Generator = np.random.default_rng(seed=None)

        #self.logger.log_args(self.params) #add to log


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

    def get_ref_path(self):
        currdir = os.path.dirname(__file__)
        
        return currdir+"/../refsolution/value.csv"

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
        arr = np.loadtxt(self.get_ref_path(),
                        delimiter=",") 
        return np.unique(arr[:,0])

    def get_reference_data(self,time):

        #time is the time you want the data

        ##fetch the reference data for halfbridge
        arr = np.loadtxt(self.get_ref_path(),
            #os.path.join(self.dirname, "./src/refsolution/value.csv"),
                        delimiter=",")  
       
        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,5))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        #RETURNS X, DU
        return arrtemp[:,1],arrtemp[:,-1]

    def du_ref(self,q,time):

        #returns the drift of the reference solution 
        qref,duref = self.get_reference_data(time)
        return np.interp(q,qref.reshape(5000),duref.reshape(5000),left=duref[0],right=duref[-1])

    def gfun(self,q):
        #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U

        return (self.Ufinal(q) - (q**2)/2)
 
    def f(self,q,t):

        return 0

    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):
        #takes one step forward in the euler maruyama discretisation
        
        return xn + (self.sigma0*du_bar)*dt + self.sigma0*dw



class LinearOU():
    def __init__(self,d=1):
        self.d = d #dimension #TODO!!! this needs to be added 
        self.gamma = 1
        self.A = 0.2
        self.sigma0 = 1
        self.Lambda = 1

        #get params 
        params = boundary_params()
        self.epsilon = params.epsilon 
        self.Tf = params.Tf
        self.g = params.g

        #set up npr generator 
        self.Generator = np.random.default_rng(seed=None)

        
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
        return -self.sigma0*np.exp(self.A*(self.Tf-time))*self.gamma


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

    def get_reference_timegrid(self):
         
        return None

    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):
        #takes one step forward in the euler maruyama discretisation
        
        return xn + (self.b(xn,t) \
                     + self.sigma0*du_bar)*dt + self.sigma0*dw



class QuadraticOU():
    def __init__(self,Tf,d=1):
        self.d = d #dimension #TODO!!! this needs to be added 
        self.Tf = Tf
        self.A = 0.2
        self.P = 0.2 
        self.Q = 0.1
        self.sigma0 = 1
        self.Lambda = 1

        #set up npr generator 
        self.Generator = np.random.default_rng(seed=None)

        
    def q_marginal_initial(self,q):

        return np.exp(-(q**2))/np.sqrt(np.pi)

    def q_marginal_final(self,q): 
        
        return np.exp(-(q**2)/2)/np.sqrt(2*np.pi)
        #qfinal(q)/self.pf_norm

    #generate the initial data
    def generate_samples_q(self,batch_samples,mode):
        
        ##sample points from the intial distribution for q 
        #output as a tensor

        #from where to sample the q marginals
        if mode=="initial":
            out = self.Generator.normal(loc=0,scale=np.sqrt(0.5),size=shape)
        elif mode=="final": 
            out = np.zeros((batch_samples,1))
            #q_marginal = self.q_marginal_final
        
        return out

    def analytic_sol(self,time):
        return -self.sigma0*np.exp(self.A*(self.Tf-time))*self.gamma


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



class UnderdampedHalfBridge():
    def __init__(self):

        #get params 
        self.params = boundary_params()
        self.epsilon = self.params.epsilon 
        self.Tf = self.params.Tf
        self.g = self.params.g
        self.beta = 2
        self.tau = 1
        self.m = 1
        self.sigma0 = np.sqrt(2/self.beta)

        self.Generator = np.random.default_rng(seed=None)

        self.pf_norm = np.trapz(self.qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        self.pi_norm = np.trapz(self.qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))

    def Uinitial(self,q):

        return (q**4)/4

    def Ufinal(self,q):

        return ((q**2-1)**2)/4

    def scorefinal(self,q):

        return -(q**2-1)*q

    def qinitial(self,q):

        return np.exp(-self.beta*self.Uinitial(q))

    def qfinal(self,q):

        return np.exp(-self.beta*self.Ufinal(q)) 

    def get_ref_path(self):
        #currdir = os.path.dirname(__file__)
        print("Reference solution not available.")
        return  #currdir+"/../refsolution/value.csv"

    def q_marginal_initial(self,q):

        return self.qinitial(q)/(self.pi_norm)


    def q_marginal_final(self,q): 
        
        return self.qfinal(q)/self.pf_norm


    def p_initial(self,q,p):

        return (self.q_marginal_initial(q)*np.exp(-self.beta*(p**2)/2))/np.sqrt(2*np.pi/self.beta)


    def p_final(self,q,p): 
        
        return (self.q_marginal_final(q)*np.exp(-self.beta*(p**2)/2))/np.sqrt(2*np.pi/self.beta)

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
    
    def get_reference_timegrid(self):
        #arr = np.loadtxt(self.get_ref_path(),
        #                delimiter=",") 
        return np.linspace(0,self.Tf)

    def get_reference_data(self,time):

        #time is the time you want the data

        ##fetch the reference data for halfbridge
        
        ##fetch the reference data approximation data
        #arrtemp = arr[np.where(arr[:,0]==np.round(time,5))] 
        #arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        #RETURNS X, DU for plotting
        return np.zeros(1000),np.zeros(1000)

    def du_ref(self,q,time):

        #returns the drift of the reference solution 
        #print("Reference solution not available.")
        return np.zeros_like(q)

    def gfun(self,x):

        #Terminal cost
        return -self.beta*((x[:,1,None]**2)/2 + self.Ufinal(x[:,0,None]))

    def f(self,x,t):

        return 0

    def b(self,x,t):

        return 0
    
    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):

        #takes one step forward in the euler maruyama discretisation
        #go forward one step 
        xn[:,0,None] = xn[:,0,None] + (xn[:,1,None]/self.m)*dt 
        xn[:,1,None] = xn[:,1,None] - (xn[:,1,None]/self.tau + du_bar)*dt \
                                    + self.sigma0*dw

        return xn

class UnderdampedEntropyProduction(UnderdampedHalfBridge):
   
    def euler_maruyama_step(self,xn,du_bar,dt,dw,t):

        #takes one step forward in the euler maruyama discretisation
        #go forward one step 
        xn[:,0,None] = xn[:,0,None] + (xn[:,1,None]/self.m - (self.g*self.tau/self.m)*du_bar)*dt \
                                    + np.sqrt(2*self.tau*self.g/(self.m*self.beta))*dw[:,0,None]
        xn[:,1,None] = xn[:,1,None] - (xn[:,1,None]/self.tau + du_bar)*dt \
                                    + self.sigma0*dw[:,1,None]

        return xn


    def f(self,x,t,du_bar):

        return (self.beta/(self.tau*self.m))*(x[:,1,None]**2) - np.gradient()


if __name__=="__main__":

    boundary = UnderdampedHalfBridge()
    temp = boundary.generate_samples(100,"initial")
    print(temp[:,0,None].shape)
    