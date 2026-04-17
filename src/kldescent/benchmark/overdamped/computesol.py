import os
import warnings #ignore divide by zeros for the boundaries

import numpy as np

from kldescent.benchmark.heatintegrators import *
from kldescent.benchmark.overdamped.iteration import CellProblem
from kldescent.benchmark.utilfunctions import * 

def compute_rho(phi_t,hatphi_t):
    
    return phi_t*hatphi_t

def compute_sigma(phi_t,A,bridge_type):
    
    with warnings.catch_warnings(): #ignore log(0) error as this is handled later
        warnings.simplefilter("ignore") 
        if bridge_type=="half":
            
            return -2*np.sqrt(A)*np.nan_to_num(np.log(phi_t),posinf=0,neginf=0,nan=0)
        else:

            return 0#-(A/B)*np.gradient(phi_t,q)/phi_t

def compute_dV(q,hatphi_t,phi_t,A,B,bridge_type):
    
    with warnings.catch_warnings(): #ignore log(0) error as this is handled later
        warnings.simplefilter("ignore") 
        if bridge_type=="half":
            
            dphi_t = np.gradient(phi_t,q)
            dhatphi_t = np.gradient(hatphi_t,q)

            term1 = -dphi_t/(np.sqrt(A)*phi_t)
            term2 = -B*np.sqrt(A)*np.trapezoid(hatphi_t*dphi_t,q)/(A*(A-B))
            term3 = ((1/np.sqrt(A)) - 1)*(hatphi_t*dphi_t + phi_t*dhatphi_t)/(2*hatphi_t*phi_t)
            
            return np.nan_to_num(term1 + term2 + term3,posinf=0,neginf=0,nan=0)
        else:

            return -(A/B)*np.gradient(phi_t,q)/phi_t

def save_solution_in_time(cell,filename_full,integrator):
    """
    Evaluate the schrödinger factors along the time horizon and use them to compute 
    the probability distribution and value function in the horizon [0,Tf2]
    
    args: 
        cell : class
            class containing solution of the cell problem
        filename : str
            location of csv where results will be saved
        integrator : function
            method to use for integrating the heat equations
        
    returns:
        arr_out : arr
            array of shape (n,4) with columns containing [t,q,rho(t,q),V(t,q)]
    """

    #define a grid for q
    q = np.linspace(-cell.qmax,cell.qmax,num=cell.num_samples)

    #define timepoints to save solution 
    #time points in the slow time (t2)
    tvec = np.linspace(0,cell.Tf2,num=cell.num_tsteps)
    dt = tvec[1] 

    #iterate over times to compute the solution at each timepoint
    xhatphi_t,hatphi_t = cell.xpsi.copy(),cell.hatphi0.copy()
    xphi_t = cell.xpsi.copy() 
    phi_t = cell.psiT.copy() #phi 0 

    for t in range(cell.num_tsteps):

        #interpolate (skip for MC method)
        #phi_t = interpolate_(q,xphi_t,phi_temp)
        #hatphi_t = interpolate_(q,xhatphi_t,hatphi_temp)       
        
        #apply convolution filters for smoothing
        #phi_t = apply_filter(phi_t)
        #hatphi_t = apply_filter(hatphi_t)
        phi_t = np.nan_to_num(phi_t,posinf=0,neginf=0,nan=0)
        hatphi_t = np.nan_to_num(hatphi_t,posinf=0,neginf=0,nan=0)
        
        arr_temp = np.column_stack((tvec[t]*np.ones(cell.num_samples),q,
                                        normalise(q,compute_rho(phi_t,hatphi_t)),
                                        compute_sigma(phi_t,cell.A,cell.bridge_type)))
        
        arr_temp = zchop(arr_temp, tol=1e-6)

        append_array_to_csv(filename_full,arr_temp)#[arr_temp[:,-1]!=0])

        del arr_temp 

        #compute next step of the hatphi and phi evolutions 
        xhatphi_t,hatphi_t = integrator(xhatphi_t,hatphi_t,dt,cell.beta,num_steps=np.minimum(1,int(np.floor(1000/cell.num_tsteps))))
        xphi_t,phi_t = integrator(cell.xpsi.copy(),cell.psi0.copy(),cell.Tf2-(t+1)*dt,cell.beta,num_steps=np.minimum(1,int(np.floor((cell.num_tsteps-t)*1000/cell.num_tsteps))))

    return 

def compute_overdamped(boundary,save_location,bridge_type="half"):
    """Compute the overdamped cell problem
    params:
        boundary : class
            defines the boundary conditions
        savelocation : str
            location where to save csv
        bridgetype : str 
            determines the type of the cell problem
        
    returns:
        csv saved in savelocation
    """

    #make folder for results if not exists
    os.makedirs(save_location,exist_ok=True)

    cell = CellProblem(boundary,save_location,bridge_type=bridge_type)
    cell.iteration(integrator=mcintegrator,
                    max_iters=20)

    filename_full = make_csv_and_header(save_location,filename="overdamped",header="#t,q,rho,sigma")
    save_solution_in_time(cell,
                                filename_full,
                                integrator=mcintegrator)

    #arr  = zchop(arr,tol=1e-6)
    
    #save_array(arr,save_location,filename="overdamped",header="t,q,rho,dV")

    return 

if __name__=="__main__":
    from ...sde.boundary import UnderdampedGaussian,UnderdampedLandauer
    from kldescent.utils.parser import fetch_results_folder_from_cmd
    boundary = UnderdampedLandauer()
    folder = fetch_results_folder_from_cmd() 
    compute_overdamped(boundary,folder,bridge_type="half")
    
    