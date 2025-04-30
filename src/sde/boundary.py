import numpy as np 
import os
#the boundary conditions    

def qinitial(q):

    return (np.exp(-(q**4)/4))

def qfinal(q):

    return np.exp(-((q**2-1)**2)/4) 

class HalfBridgeLandauer():
    def __init__(self):
        self.pf_norm = np.trapz(qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        #self.pi_norm = np.trapz(qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))

    def get_ref_path(self):
        currdir = os.path.dirname(__file__)
        
        return currdir+"/../refsolution/value.csv"

    def q_marginal_initial(self,q):

        return #qinitial(q)/(self.pi_norm)



    def q_marginal_final(self,q): 
        
        return qfinal(q)/self.pf_norm

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
                            
        #torch.tensor(,dtype=torch.float32).reshape((batch_samples,1))
        
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
        arrtemp = arrtemp[arrtemp[:,1].argsort()] #sort by x

        #RETURNS X, DU
        return arrtemp[:,1],arrtemp[:,-1]



class LinearOU():
    def __init__(self):
        return

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
            q_marginal = self.q_marginal_initial
        elif mode=="final": 
            q_marginal = self.q_marginal_final

        probs = q_marginal(np.linspace(-8,8,10000))/np.sum(q_marginal(np.linspace(-8,8,10000)))
        
        
        return np.random.choice(np.linspace(-8,8,10000),
                            shape,
                            p=probs)
                            
        #torch.tensor(,dtype=torch.float32).reshape((batch_samples,1))

    def get_reference_data(self,time):

        #time is the time you want the data

        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,5))]
        arrtemp = arrtemp[arrtemp[:,1].argsort()] #sort by x

        #RETURNS X, DU
        return arrtemp[:,1],arrtemp[:,-1]

        return         