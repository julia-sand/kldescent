import numpy as np 

#the boundary conditions    
#TODO: check best place for these to reduce recomputation.
pi_norm = np.trapz(np.exp(-(np.linspace(-8,8,1000)**4)/4),np.linspace(-8,8,1000))
pf_norm = np.trapz(np.exp(-((np.linspace(-8,8,1000)**2-1)**2)/4),np.linspace(-8,8,1000))


def q_marginal_initial(q):

    return (np.exp(-(q**4)/4))/(pi_norm)


def q_marginal_final(q): 
    
    return (np.exp(-((q**2-1)**2)/4))/(pf_norm)

def p_initial(q,p):

    return (q_marginal_initial(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)


def p_final(q,p): 
    
    return (q_marginal_final(q)*np.exp(-(p**2)/2))/np.sqrt(2*np.pi)

