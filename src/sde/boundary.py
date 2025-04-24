import numpy as np 

#the boundary conditions    

def qinitial(q):

    return (np.exp(-(q**4)/4))

def qfinal(q):

    return np.exp(-((q**2-1)**2)/4) 

class AssignedBoundaryData():
    def __init__(self):
        self.pf_norm = np.trapz(qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))
        self.pi_norm = np.trapz(qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))

    def q_marginal_initial(self,q):

        return qinitial(q)/(self.pi_norm)



    def q_marginal_final(self,q): 
        
        return qfinal(q)/self.pf_norm

    def p_initial(self,q,p):

        return (self.q_marginal_initial(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)


    def p_final(self,q,p): 
        
        return (self.q_marginal_final(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)

