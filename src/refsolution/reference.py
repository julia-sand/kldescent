import os

import numpy as np 

from ..sde.boundary import *
from .overdamped.gauss import Gaussian_Overdamped_Ref
from .utilfunctions import *
from .perturbative.driftinterp import perturbative_du_interp
from .perturbative.marginalhalf import MarginalHalfBridgePerturbation
from .overdamped.sol import compute_overdamped

class RefSolution():
    """
    Class used for organising the location
    for the reference solutions used in plotting
    
    Methods:
        get_reference_timegrid()
            Fetches time discretisation from reference solution
        get_reference_data()
            fetches reference data found in ref sol path
        du_ref()
            reference drift for integration
    """

    def __init__(self, boundary=None, folder_name=None):
        """
        Initialises parameters and directs to the reference solutions. 
        If no reference solution is available in the folder, then computes the reference solution
        args:
            boundary : class
                specifies the boundary conditions
            folder_name : str
                if no boundary condition specified, the folder can be set manually using this parameter. 
        """
        self.boundary= boundary

        if folder_name is not None: #use this if running the file directly
            self.folder_name = folder_name
        else:
            self.folder_name = f"{self.boundary.__class__.__name__}"
                
        os.makedirs(self.get_ref_path(),exist_ok=True)

        #sets reference file locations
        self.overdamped_loc = self.get_ref_path()+"/overdamped.csv"
        self.perturb_underdamped_loc = self.get_ref_path()+"/perturbative_underdamped.csv"
        self.ref_loc = self.get_ref_path()+"/refsol.csv"

        try:
            #compute corrections
            self.duref_net = perturbative_du_interp(self.boundary,self)
        except FileNotFoundError:
            print("Overdamped results not available. Solving cell problem...")
            #compute overdamped ref solution
            compute_overdamped(self.boundary,self.get_ref_path())
            print("Cell problem completed. Computing Corrections...")
            pertsol = MarginalHalfBridgePerturbation(boundary,self)
            pertsol.save_perturbative()
            #pertsol.save_perturbative_cumulants()
            print("Underdamped Results saved.")
            self.duref_net = perturbative_du_interp(self.boundary,self)
        
        self.duref_net.eval()

    def get_ref_path(self):
        #returns the folder where the reference solution is expected to be

        return os.path.dirname(__file__) + "/" + self.folder_name
    
    def get_ref_data(self,data=None):
        """"
        Returns an array for reference solution is expected to be
        """
        
        if data == "overdamped":
            arr = np.loadtxt(open(self.overdamped_loc),
                        delimiter=",",skiprows=1) 
        elif data == "perturbative underdamped":
            arr = np.loadtxt(open(self.perturb_underdamped_loc),
                        delimiter=",",skiprows=1) 
        else: 
            arr = np.loadtxt(open(self.ref_loc),
                        delimiter=",",skiprows=1) 
        return arr 

    def get_reference_timegrid(self,time=None):
        try: 
            arr = self.get_ref_data()
            return np.unique(arr[:,0])
        except FileNotFoundError:
            return np.linspace(0,time)
    
    def get_reference_du(self,time):

        #time is the time you want the data

        ##fetch the reference data for halfbridge
        arr = self.get_ref_data()  
       
        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,8))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        #RETURNS X, DU
        return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,-1])

    def get_quantity(self,time,quantity,data=None,arr=None):

        #time is the time you want the data

        ##fetch the reference data for halfbridge
        if arr is None: #default overdamped
            arr = self.get_ref_data(data=data)  
       
        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,8))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        if quantity =="value":
            idx=3
        elif quantity =="marginal":
            idx=2
        elif quantity =="underdamped drift":
            idx=4
            
        #RETURNS X, DU
        return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,idx])

    def du_ref(self,q,time):

        #returns the drift of the reference solution 
        try:
            qref,duref = self.get_ref_drift_gaussian(time)
            return np.interp(q,qref,duref,left=duref[0],right=duref[-1])
        except:
            return np.zeros_like(q)



class GaussRefSolution(RefSolution):
    """
    Extends the base class adding analytic solutions for gaussian boundary conditions
    """

    def __init__(self,boundary=None,folder_name=None):   
        
        RefSolution.__init__(boundary,folder_name)
        
        #only for gaussian
        self.refsol_phi = Gaussian_Overdamped_Ref(self.boundary)

        #locate the cumulants 
        self.cumulants_loc = self.get_ref_path()+"/refcumulants.csv"   


    def get_ref_cumulants_file(self):
        #returns the folder where the reference solution is expected to be
        arr = np.loadtxt(open(self.cumulants_path),
                        delimiter=",",skiprows=1) 
        return arr 

    def get_ref_drift_gaussian(self,time):
        arr = self.get_ref_data()
        
        return np.linspace(-4,4),-(np.squeeze(arr[np.where(arr[:,0]==time),-1]) + np.squeeze(arr[np.where(arr[:,0]==time),-2])*np.linspace(-4,4))

    def get_known_solution_cumulants(self,cumulant_type):
        arr = self.get_ref_cumulants_file() 
        try:  
            if cumulant_type == "Position Variance": 
                temp = np.squeeze(arr[:,1])#np.squeeze(arr[:,4])
            elif cumulant_type == "Momentum Variance":
                temp = np.squeeze(arr[:,3])#np.squeeze(arr[:,6])
            elif cumulant_type == "Cross Correlation":
                temp = np.squeeze(arr[:,2])#np.squeeze(arr[:,5])
            elif cumulant_type == "Momentum Mean":
                temp = np.squeeze(arr[:,8])
            elif cumulant_type == "Position Mean":
                temp = np.squeeze(arr[:,7])
            else:
                temp = np.zeros_like(np.squeeze(arr[:,0]))
        except IndexError:
            return np.zeros_like(np.squeeze(arr[:,0])),np.zeros_like(np.squeeze(arr[:,0]))

        return np.squeeze(arr[:,0]),temp

def plot_data(reference_,quantity,data):
    
    times = [0,0.06,0.12,0.2]

    fig,axs = plt.subplots(ncols=len(times),nrows=1)
    fig.set_figheight(15)
    fig.set_figwidth(25)

    for time in enumerate(times):
        q, ptq = reference_.get_quantity(time[1], quantity, data=data)
        axs[time[0]].plot(q,ptq)
        axs[time[0]].set_title(f"t={time[1]}")
        
    fig.tight_layout()

    fig.savefig(reference_.get_ref_path()+f"/{data}_{quantity}.png",bbox_inches="tight")        

if __name__=="__main__":
    import matplotlib.pyplot as plt
    boundary = UnderdampedLandauer()
    reference_ = RefSolution(boundary)
    plot_data(reference_,quantity="value",data="overdamped")
    plot_data(reference_,quantity="marginal",data="overdamped")
    plot_data(reference_,quantity="underdamped drift",data="perturbative underdamped")
