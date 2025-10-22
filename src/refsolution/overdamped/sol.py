import numpy as np

import src.sde.boundary as boundaries
import proximal,iteration

#Proximal operator solution for fokkerplanck equation
def afun(Tf):
    return (Tf - 1 + np.exp(-Tf))/Tf

def bfun(Tf):
    return 1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf)

def compute_rho(phi_t,hatphi_t):
    
    return phi_t*hatphi_t

def compute_V(phi_t,A,B):
    
    return -(A/B)*np.log(phi_t)

def interpolate_(q,xs,phis):
    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx])

def interpolate_(q,xs,phis):
    idx = np.argsort(xs)

    return np.interp(q,xs[idx],phis[idx])

def normalise(x,y):
    idx = np.argsort(x)
    return y/np.trapz(y[idx],x[idx])

def get_solution_in_time(xpsi,
                        psiT,Tf2,A,B,
                        xnum = 1000,
                        tnum = 50):
    """
    Evaluate the schrödinger factors along the time horizon and use them to compute 
    the probability distribution and value function in the horizon [0,Tf2]
    
    args: 
        xpsi : arr
            position coordinates returned by iteration
        psiT : arr
            densities returned by iteration, equivalent to phi0 in paper notation
        Tf2 : float
            time horizon for the cell problem
        A : float
            constant defining the cell problem
        B : float
            constant defining the cell problem 
        xnum : int
            number of coordinates in the space discretisation 
        tnum : int
            number of coordinates in the time discretisation

    returns:
        arr_out : arr
            array of shape (n,4) with columns containing [t,q,rho(t,q),V(t,q)]
    """

    #define a grid for q
    q = np.linspace(-4,4,num=xnum)

    #define timepoints to save solution 
    #time points in the slow time (t2)
    tvec = np.linspace(0,Tf2,num=tnum)
    dt = tvec[1] 

    #set hatphi initial conditions from psi
    hatphi0 = boundary.q_marginal_initial(xpsi)/psiT
    xhatphi = xpsi.copy()

    #preallocate arrays
    arr_out = np.array((0,4))

    #iterate over times to compute the solution at each timepoint
    for t in tvec:

        #compute one step of the proximal recursion
        xhatphi_t,hatphi_temp = proximal.proximal_fpk(xhatphi,hatphi0,dt,1/A,num_steps=int(np.floor(10000/tnum)))
        xphi_t,phi_temp = proximal.proximal_fpk(xpsi,psiT,t,1/A,num_steps=int(np.floor(10000/tnum)))
        
        #interpolate 
        phi_t = interpolate_(q,xphi_t,phi_temp)
        hatphi_t = interpolate_(q,xhatphi_t,hatphi_temp)
        
        #compute rho and V
        arr_out = np.append(arr_out,
                    np.array([[np.ones(tnum*xnum)*t,
                                q,
                                normalise(q,compute_rho(phi_t,hatphi_t)),
                                compute_V(phi_t)]]),axis=0)#cross corr
           
        #set new iterates
        xhatphi, hatphi0 = xhatphi_t,hatphi_temp 
        xpsi, psiT = xphi_t,phi_temp

    return arr_out

def save_array(arr_out,boundary):
    #make dir
    os.makedirs(f"src/refsolution/{boundary.__class__.__name__}",exist_ok=True)
    print(f"src/refsolution/{boundary.__class__.__name__}")
    np.savetxt(boundary.ref_path, arr_out ,fmt="%10.6f",header = "t,q,rho,V",delimiter=',')
    print(boundary.ref_path)
    return

if __name__=="__main__":
    boundary = boundaries.UnderdampedGaussian()
    Tf = boundary.params["Tf"]
    Tf2 = boundary.params["Tf"]*(boundary.params["epsilon"]**2)
    
    #should these be in the boundary class? 
    A = afun(Tf)
    B = bfun(Tf)

    xhatphi,xpsi,hatphiT,psiT = iteration.iteration(boundary,A,B,Tf2)

    arr = get_solution_in_time(xpsi,
                        psiT,Tf2,A,B,
                        xnum = 1000,
                        tnum = 50)
    save_array(arr,boundary)