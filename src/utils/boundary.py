import numpy as np 

#the boundary conditions    
#TODO: check best place for these to reduce recomputation.



def qinitial(q):

    return (np.exp(-(q**4)/4))

pi_norm = np.trapz(qinitial(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))

def q_marginal_initial(q):

    return qinitial(q)/(pi_norm)

def qfinal(q):

    return np.exp(-((q**2-1)**2)/4) #+ ((q**2)/2))

pf_norm = np.trapz(qfinal(np.linspace(-8,8,1000)),np.linspace(-8,8,1000))

def q_marginal_final(q): 
    
    return qfinal(q)/pf_norm

def p_initial(q,p):

    return (q_marginal_initial(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)


def p_final(q,p): 
    
    return (q_marginal_final(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)

