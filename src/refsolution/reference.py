import os

import numpy as np 

from ..sde.boundary import *
from ..utils.parser import fetch_results_folder_from_cmd
from .overdamped.gauss import Gaussian_Overdamped_Ref
from .utilfunctions import *
from .perturbative.driftinterp import perturbative_du_interp
from .perturbative.marginalhalf import MarginalHalfBridgePerturbation
from .overdamped.sol import compute_overdamped
from ..model.nn import MLP,NeuralNetwork

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

    def __init__(self, boundary=None, folder_name=None, bridge_type="marginal"):
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
            self.folder_name = fetch_results_folder_from_cmd() #f"{self.boundary.__class__.__name__}"
                
        os.makedirs(self.get_ref_path(),exist_ok=True)

        #sets reference file locations
        self.overdamped_loc = self.get_ref_path()+"/overdamped.csv"
        self.perturb_underdamped_loc = self.get_ref_path()+"/perturbative_underdamped.csv"
        self.ref_loc = self.get_ref_path()+"/refsol.csv"

        #initialise a simple neural net
        mlp_net = NeuralNetwork(self.boundary) #MLP()

        #get perturbative sol 
        try:
            #compute corrections
            self.duref_net = perturbative_du_interp(self,mlp_net)
        except FileNotFoundError:

            try: 
                print("Perturbative results not available. Computing corrections...")

                #initialise correct bridge type
                pertsol = self.get_perturbative_calculations(bridge_type)
                pertsol.save_perturbative()
            
            except FileNotFoundError:
                print("Overdamped results not available. Solving cell problem...")
                #compute overdamped ref solution

                compute_overdamped(self.boundary,self.get_ref_path())

                print("Cell problem completed. Computing Corrections...")
                
                #initialise correct bridge type
                pertsol = self.get_perturbative_calculations(bridge_type)
                
                #pertsol = MarginalHalfBridgePerturbation(boundary,overdamped_arr = self.reference.get_ref_data(data="overdamped"),)
                pertsol.save_perturbative()
                print("Perturbative Results saved.")
                
            self.duref_net = perturbative_du_interp(self,mlp_net)
        
        self.duref_net.eval()

    def get_perturbative_calculations(self,bridge_type):
        """initialise the class to compute the perturbative solution 
        depending on the bridge type
        args: 
            bridge_type : str
                "marginal", "half" 
        """

        overdamped_arr = self.get_ref_data(data="overdamped")

        if bridge_type == "marginal":
            pertsol = MarginalHalfBridgePerturbation(self.boundary,overdamped_arr,self.get_ref_path())
        else:  #update this to compute the half bridge solution
            pertsol = MarginalHalfBridgePerturbation(self.boundary,overdamped_arr,self.get_ref_path())

        return pertsol

    def get_ref_path(self):
        #returns the folder where the reference solution is expected to be

        return self.folder_name # os.path.dirname(__file__) + "/" + self.folder_name
    
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
        arr = self.get_ref_data()
        return np.unique(arr[:,0])
        
    def get_quantity(self,time,quantity,data=None,arr=None,time_type="t2"):
        """
        args:
            time : float
                the time to lookup the data
            quantity : str
                what quantity to return from the data table. Choices are: value (overdamped value function); 
                marginal (overdamped distribution); underdamped drift (perturbative estimate of underdamped DU);
                and underdamped marginal (perturbative underdamped marginal of the position)
            data : str 
                which dataset to use. Choose from overdamped (cell problem solution) or underdamped (saved perturbative solution)        
            arr : array
                if the dataset is already loaded in the script, this can be used to return the correct result
        returns:
            x : arr 
                array of shape (n,) of positions
            y : arr
                specified quantity evaluated at points x, with shape (n,)
        """

        ##fetch the reference data for halfbridge
        if arr is None: #default overdamped
            arr = self.get_ref_data(data=data)  

        if time_type=="t0":
            #multiply by eps**2 to get t2 from t0 
            time *= self.boundary.params["epsilon"]**2

        ##fetch the reference data approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,6))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x

        if quantity =="value":
            idx=3
        elif quantity =="marginal":
            idx=2
        elif quantity =="underdamped drift":
            idx=-1
        elif quantity =="underdamped marginal":
            idx=-2
        
        return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,idx]) #((0.2**2)*np.squeeze(arrtemp[:,idx]))+np.squeeze(arrtemp[:,2])

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
