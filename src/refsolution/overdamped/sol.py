import os

import numpy as np

from ...sde.boundary import UnderdampedGaussian,UnderdampedLandauer
from ..heatintegrators import *
from .iteration import CellProblem
from ..utilfunctions import * 

#Proximal operator solution for fokkerplanck equation
def compute_rho(phi_t,hatphi_t):
    
    return phi_t*hatphi_t

def compute_V(phi_t,A,B):
    
    return -(A/B)*np.log(phi_t)


def get_solution_in_time(cell,integrator,
                            xnum = 1000,
                            tnum = 11):
    """
    Evaluate the schrödinger factors along the time horizon and use them to compute 
    the probability distribution and value function in the horizon [0,Tf2]
    
    args: 
        cell : class
            class containing solution of the cell problem
        integrator : function
            method to use for integrating the heat equations
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
    tvec = np.linspace(0,cell.Tf2,num=tnum)
    dt = tvec[1] 

    hatphi0 = cell.hatphi0

    #preallocate arrays
    arr_out = np.empty((tnum*xnum,4))

    #preallocate for phi 
    phi_out = np.empty((tnum,len(cell.xhatphi),2))
    phi_out[0,:,0] = cell.xhatphi
    phi_out[0,:,1] = cell.psi0  
    

    for t in range(1,tnum):

        #compute one step of the proximal recursion
        xphi_Tt,phi_Ttemp = integrator(phi_out[t-1,:,0],phi_out[t-1,:,1],dt,cell.beta,num_steps=int(np.floor(100/tnum)))
        
        #save phi_(T-t)
        phi_out[t,:,0] = xphi_Tt
        phi_out[t,:,1] = phi_Ttemp


    #iterate over times to compute the solution at each timepoint
    xhatphi_t,hatphi_temp = cell.xpsi.copy(),cell.hatphi0.copy()
    for t in range(tnum):

        #interpolate 
        phi_t = interpolate_(q,phi_out[-t-1,:,0],phi_out[-t-1,:,1])
        hatphi_t = interpolate_(q,xhatphi_t,hatphi_temp)       

        arr_out[t*xnum:(t+1)*xnum] = np.column_stack((tvec[t]*np.ones(xnum),q,
                                                            normalise(q,compute_rho(phi_t,hatphi_t)),
                                                            compute_V(phi_t,cell.A,cell.B)))

        #compute one step of the proximal recursion
        xhatphi_t,hatphi_temp = integrator(xhatphi_t,hatphi_temp,dt,cell.beta,num_steps=int(np.floor(1000/tnum)))
             
    return np.nan_to_num(arr_out,nan=0,posinf=0,neginf=0)

def save_array(arr_out,save_location):
    
    #make dir
    #os.makedirs(save_location,exist_ok=True)

    np.savetxt(save_location + "/overdamped.csv", arr_out ,fmt="%10.6f",header = "t,q,rho,V",delimiter=',')
    return

def compute_overdamped(boundary,save_location,bridge_type="marginal half"):
    
    cell = CellProblem(boundary,save_location,bridge_type=bridge_type)
    cell.iteration(integrator=mcintegrator,
                    num_iters=25)

    arr = get_solution_in_time(cell,integrator=mcintegrator)

    save_array(arr,save_location)

if __name__=="__main__":
    boundary = UnderdampedLandauer()
    compute_overdamped(boundary,save_location,bridge_type="marginal half")
    
    