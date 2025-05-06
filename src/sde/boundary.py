import numpy as np 
import os
#the boundary conditions    


class HalfBridgeLandauer():
    def __init__(self):
        self.sigma0 = 1
        self.pf_norm = np.trapz(self.qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        #self.pi_norm = np.trapz(self.qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))


    def Uinitial(self,q):

        return (q**4)/4

    def Ufinal(self,q):

        return ((q**2-1)**2)/4

    def scorefinal(self,q):

        return (q**2-1)*q

    def qinitial(self,q):

        return np.exp(-self.Uinitial(q))

    def qfinal(self,q):

        return np.exp(-self.Ufinal(q)) 

    def get_ref_path(self):
        currdir = os.path.dirname(__file__)
        
        return currdir+"/../refsolution/value.csv"

    def q_marginal_initial(self,q):

        return #qinitial(q)/(self.pi_norm)


    def q_marginal_final(self,q): 
        
        return self.qfinal(q)/self.pf_norm


    def p_initial(self,q,p):

        return (self.q_marginal_initial(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)


    def p_final(self,q,p): 
        
        return (self.q_marginal_final(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)

    #generate the initial data
    def generate_samples_q(self,shape,mode):
        
        ##sample points from the intial distribution for q 
        #output as a tensor

        #from where to sample the q marginals
        if mode=="initial":
            out = np.zeros(shape)
            #q_marginal = self.q_marginal_initial

        elif mode=="final": 
            q_marginal = self.q_marginal_final

            probs = q_marginal(np.linspace(-8,8,10000))/np.sum(q_marginal(np.linspace(-8,8,10000)))
            out = np.random.choice(np.linspace(-8,8,10000),
                                shape,
                                p=probs)
        
        return out
                            
       
    #generate the initial data
    def generate_samples(self,batch_samples,mode):
        
        ##sample points from the intial distribution for q 
        ##and from normal distribution for p and stack together 
        #output as a tensor

        #gives the initialisation of the underdamped
             
        return torch.column_stack((self.generate_samples_q(batch_samples,mode)
                                ,torch.randn((batch_samples,1),dtype=torch.float32)))
        
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
        qref,duref = get_reference_data(self,time)
        return np.interp(q,qref.reshape(5000),duref.reshape(5000),left=duref[0],right=duref[-1])

    def g(self,q):
        #we use the -log of the final distribution. See Prop. 3 in ref for details. This is function U

        return (-np.log(self.q_marginal_final(q)) - (q**2)/2)

    def f(self,q,t):

        return 0

    def b(self,q,t):

        return 0


class LinearOU():
    def __init__(self,Tf,d=1):
        self.d = d #dimension #TODO!!! this needs to be added 
        self.Tf = Tf
        self.gamma = 1
        self.A = 0.2
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
    def generate_samples_q(self,shape,mode):
        
        ##sample points from the intial distribution for q 
        #output as a tensor

        #from where to sample the q marginals
        if mode=="initial":
            out = self.Generator.normal(loc=0,scale=np.sqrt(0.5),size=shape)
        elif mode=="final": 
            out = np.zeros(shape)
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


    def g(self,q):

        return self.gamma*q

    def f(self,q,t):

        return 0

    def b(self,q,t):

        return self.A*q


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
    def generate_samples_q(self,shape,mode):
        
        ##sample points from the intial distribution for q 
        #output as a tensor

        #from where to sample the q marginals
        if mode=="initial":
            out = self.Generator.normal(loc=0,scale=np.sqrt(0.5),size=shape)
        elif mode=="final": 
            out = np.zeros(shape)
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


    def g(self,q):

        return self.gamma*q

    def f(self,q,t):

        return 0

    def b(self,q,t):

        return self.A*q