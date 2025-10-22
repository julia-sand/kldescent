import numpy as np

def get_transient_factors(q,xpsi,psiT,t):
    """
    Evaluate the schrödinger factors along the time horizon 
    args: 
        t : float
            time to compute the factor
    returns:
        phi_t : arr
        hatphi_t : arr
    """

    #evolve hatphi0 to hatphi_t
    #get hatphi0 
    hatphi0 = boundary.q_marginal_initial(xpsi)/psiT
    #returns the factor at time t evolved by proximal operator
    xhatphi_t,hatphi_t = proximal.proximal_fpk(xpsi,hatphi0,t,1/A)

    #get phi_t = psi_(T-t)
    #returns the factor at time t evolved by proximal operator
    xphi_t,phi_t = proximal.proximal_fpk(xpsi,psiT,t,1/A)
    
    return interpolate_(q,xphi_t,phi_t),interpolate_(q,xhatphi_t,hatphi_t)

def compute_rho(phi_t,hatphi_t):
    
    return phi_t*hatphi_t

def compute_V(phi_t,A,B):
    
    return -(A/B)*np.log(phi_t)

def get_solution_in_time(xpsi,psiT,Tf2,A,B):
    
    #define a grid for q
    q = np.linspace(-4,4,)

    #define timepoints to save solution 
    #time points in the slow time (t2)
    tvec = np.linspace(0,Tf2)
    dt = tvec[1] 

    hatphi0 = boundary.q_marginal_initial(xpsi)/psiT

    #iterate over times to compute the solution at each timepoint
    for t in tvec:

        #compute one step of the proximal recursion
        xhatphi_t,hatphi_t = proximal.proximal_fpk(xpsi,hatphi0,dt,1/A)
        xphi_t,phi_t = proximal.proximal_fpk(xpsi,psiT,t,1/A)
        

        
        #compute rho and V
        rho_t = compute_rho(phi_t,hatphi_t)
        V_t = compute_V(phi_t)    
        #save to csv!!!!!

        #set new iterates
        xhatphi, hatphi0 = xhatphi_t,hatphi_t 
