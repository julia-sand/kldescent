import os
import warnings #ignore divide by zeros for the boundaries

import numpy as np

from kldescent.benchmark.heatintegrators import *
from kldescent.benchmark.overdamped.iteration import CellProblem
from kldescent.benchmark.utilfunctions import * 
from kldescent.utils.misc import append_array_to_csv,save_array


def compute_rho(phi_t,hatphi_t):
    
    return phi_t*hatphi_t

def compute_sigma(phi_t,A,bridge_type):
    
    with warnings.catch_warnings(): #ignore log(0) error as this is handled later
        warnings.simplefilter("ignore") 
        if bridge_type=="marginal":
            
            raise NotImplementedError #-2*np.sqrt(A)*np.nan_to_num(np.log(phi_t),posinf=0,neginf=0,nan=0)
        
        elif bridge_type in ("half","full"):

            return -2*np.sqrt(A)*np.nan_to_num(np.log(phi_t),posinf=0,neginf=0,nan=0) #-(A/B)*np.gradient(phi_t,q)/phi_t

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
    
    Parameters
    ----------    
    cell : object
        object containing the solution of the cell problem
    filename : str
        location of csv where results will be saved
    integrator : callable
        method to use for integrating the heat equations
    
    Returns
    -------
    arr_out : np.ndarray
        Array of shape (n, 4) where each column corresponds to:
        [t,q,rho(t,q),DU(t,q)]
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
        
        arr_temp = zchop(arr_temp, tol=1e-8)

        append_array_to_csv(filename_full,arr_temp)#[arr_temp[:,-1]!=0])

        del arr_temp 

        #compute next step of the hatphi and phi evolutions 
        xhatphi_t,hatphi_t = integrator(xhatphi_t,hatphi_t,dt,cell.beta)
        xphi_t,phi_t = integrator(cell.xpsi.copy(),cell.psi0.copy(),cell.Tf2-(t+1)*dt,cell.beta)

    return 

def compute_overdamped(boundary,save_location,bridge_type,filename="overdamped"):
    """
    Compute the overdamped cell problem
    
    Parameters
    ----------
    boundary : object
        Defines the boundary conditions of the problem.

    save_location : str
        Directory where the CSV file will be saved.

    bridge_type : str
        Determines the type of cell problem to solve. ("marginal","half", or "full")

    Returns
    -------
    None
        The function writes a CSV file to `save_location`.
    """

    #make folder for results if not exists
    os.makedirs(save_location,exist_ok=True)

    cell = CellProblem(boundary,save_location,bridge_type=bridge_type)
    cell.iteration(integrator=mcintegrator)

    filename_full = make_csv_and_header(save_location,filename,header="#t,q,rho,sigma")
    save_solution_in_time(cell,
                                filename_full,
                                integrator=mcintegrator)

    return 

def save_rescaled_csv(Tf,folder,filename="overdamped"):
    '''prepare the overdamped csv for pretraining input into the neural network

    Rescales the t2 time scale of the overdamped solution to match the Tf input

    Saves a copy with the rescaled time and areas of no probability density (rho = 0) removed  
    
    Tf : float
        final time for the rescaled solution 

    folder : str
        location of original csv and output location
    filename : str 
        filename of original csv, output filename has the word rescaled appended to it. 

    '''

    #load csv 
    arrtemp = np.loadtxt(folder+"/"+filename+".csv",
                    delimiter=",", skiprows=1)

    Tf2 = arrtemp[-1,0] #find final cell problem time
    arrtemp[:,0] *= (Tf/Tf2) 

    dx = np.abs(arrtemp[1,1]-arrtemp[0,1])
    #add filtering, consistent with the underdamped solution    
    arrtemp[:,2] = savgol_filter_(arrtemp[:,2],dx=dx,deriv=0)
    arrtemp[:,3] = savgol_filter_(arrtemp[:,3],dx=dx,deriv=0)

    #remove zero probability areas
    arr = arrtemp[np.where(arrtemp[:,2]>0)] 
    save_array(arr,folder,filename+"_rescaled",header="#t,q,rho,sigma")

    return 

if __name__=="__main__":
    from ...sde.boundary import UnderdampedLandauer
    from kldescent.utils.parser import fetch_results_folder_from_cmd
    boundary = UnderdampedLandauer()
    folder = fetch_results_folder_from_cmd() 
    compute_overdamped(boundary,folder,bridge_type="half")

    #save a copy of the overdamped solution scaled to the final time given by the boundary conditions. 
    save_rescaled_csv(boundary.params["Tf"],folder,filename="overdamped")