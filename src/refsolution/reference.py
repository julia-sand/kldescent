import os

import numpy as np 

from ..sde.boundary import *
from ..utils.parser import fetch_results_folder_from_cmd
from ..model.nn import MLP,NeuralNetwork
from .utilfunctions import *
from .perturbative.driftinterp import perturbative_du_interp
from .perturbative.corrections import MarginalHalfBridgePerturbation,HalfBridgePerturbation
from .overdamped.sol import compute_overdamped
from .overdamped.gauss import Gaussian_Overdamped_Ref
from .gaussian import GaussBVP

class BaseRefSolution():

    def __init__(self, bridge_type="half", boundary=None, folder_name=None):
        
        self.boundary = boundary

        if folder_name is not None: #use this if running the file directly
            self.folder_name = folder_name
        else:
            self.folder_name = fetch_results_folder_from_cmd() #f"{self.boundary.__class__.__name__}"
                
        os.makedirs(self.folder_name,exist_ok=True)

        #sets reference file locations
        self.overdamped_loc = self.folder_name+"/overdamped.csv"
        self.perturb_underdamped_loc = self.folder_name+"/perturbative_underdamped.csv"
        self.ref_loc = self.folder_name+"/refsol.csv"
        self.perturbative_cumulants_loc = self.folder_name+"/perturbative_cumulants.csv"

    def get_perturbative_calculations(self,bridge_type,overdamped_arr):
        """initialise the class to compute the perturbative solution 
        depending on the bridge type
        args: 
            bridge_type : str
                "marginal", "half" 
        """

        if bridge_type == "marginal":
            pertsol = MarginalHalfBridgePerturbation(self.boundary,overdamped_arr,self.folder_name())
        
        elif bridge_type == "half":  #update this to compute the half bridge solution
            
            pertsol = MarginalHalfBridgePerturbation(self.boundary,overdamped_arr,self.folder_name())
        
        else: 
        
            pertsol = HalfBridgePerturbation(self.boundary,overdamped_arr,self.folder_name())

        return pertsol

class RefSolution(BaseRefSolution):
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

    def __init__(self, bridge_type="half", boundary=None, folder_name=None):
        r"""
        Initialises parameters and directs to the reference solutions. 
        If no reference solution is available in the folder, then computes the reference solution
        args:
            bridge_type : str
                type of bridge. Choose from "marginal" or "half". 
            boundary : class
                specifies the boundary conditions
            folder_name : str
                if no boundary condition specified, the folder can be set manually using this parameter. 
        """
        
        BaseRefSolution.__init__(self,bridge_type,boundary,folder_name)
        self.initialise_net(bridge_type)

    def initialise_net(self,bridge_type):

        #initialise a simple neural net to approximate the drift 
        mlp_net = NeuralNetwork(self.boundary) #MLP()

        #get perturbative sol 
        try:
            #compute corrections
            self.duref_net = perturbative_du_interp(self,mlp_net)
        
        except BaseException:
            #train from default initialisation 
            self.duref_net = perturbative_du_interp(self,mlp_net)
        
        except FileNotFoundError:

            try: 
                #try to fetch overdamped results
                overdamped_arr = self.get_ref_data(data="overdamped")

                print("Perturbative results not available. Computing corrections...")

                #initialise correct bridge type
                pertsol = self.get_perturbative_calculations(bridge_type,overdamped_arr)
                pertsol.save_perturbative()
                pertsol.save_perturbative_cumulants()

            except FileNotFoundError:
                print("Overdamped results not available. Solving cell problem...")

                #compute overdamped ref solution
                compute_overdamped(self.boundary,self.get_ref_path(),bridge_type=bridge_type)

                print("Cell problem completed. Computing Corrections...")
                overdamped_arr = self.get_ref_data(data="overdamped")

                #initialise correct bridge type
                pertsol = self.get_perturbative_calculations(bridge_type,overdamped_arr)
                
                pertsol.save_perturbative()
                pertsol.save_perturbative_cumulants()
                print("Perturbative Results saved.")
                
            self.duref_net = perturbative_du_interp(self,mlp_net)
        
        self.duref_net.eval()

    def get_ref_path(self):
        """return the folder where the reference solution files are"""

        return self.folder_name # os.path.dirname(__file__) + "/" + self.folder_name
    
    def get_ref_data(self,data=None):
        """Return an array for reference solution"""
        
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
        Get quantities to plot for the reference solution 
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
            time_type : str
                if "t0", change it first to t2. 
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

    def get_perturbative_cumulants_array(self):
        """return the array for predicted cumulants using perturbation theory"""
        arr = np.loadtxt(open(self.perturbative_cumulants_loc),
                        delimiter=",",skiprows=1) 
        return arr 

    def get_perturbative_cumulant(self,cumulant_type):
        arr = self.get_perturbative_cumulants_array() 
        try:  
            if cumulant_type == "Position Variance": 
                temp = np.squeeze(arr[:,3])
            elif cumulant_type == "Momentum Variance":
                temp = np.squeeze(arr[:,4])
            elif cumulant_type == "Cross Correlation":
                temp = np.squeeze(arr[:,5])
            elif cumulant_type == "Momentum Mean":
                temp = np.squeeze(arr[:,2])
            elif cumulant_type == "Position Mean":
                temp = np.squeeze(arr[:,1])
            else:
                print("Perturbative prediction for "+cumulant_type+" not available.")
                temp = np.zeros_like(np.squeeze(arr[:,0]))
        except IndexError:
            print("Perturbative prediction for "+cumulant_type+" not available.")
            return np.zeros_like(np.squeeze(arr[:,0])),np.zeros_like(np.squeeze(arr[:,0]))

        return np.squeeze(arr[:,0]),temp


class GaussRefSolution(BaseRefSolution):
    """Compile the gaussian versions of the reference solutions"""

    def __init__(self,bridge_type="half",boundary=None,folder_name=None):   
        
        BaseRefSolution.__init__(self,bridge_type="half",boundary=boundary,folder_name=folder_name)
        
        #only for gaussian
        self.refsol_phi = Gaussian_Overdamped_Ref(self.boundary)

        #set different filename for underdamped cumulants 
        self.underdamped_cumulants_loc = self.folder_name+"/underdamped_cumulants.csv"
        self.underdamped_loc = self.folder_name+"/underdamped.csv"
        self.duref_net = None

        #self.solve_bvp_gaussian()

    def solve_bvp_gaussian(self,):

        #solve BVP for gaussian 
        bvp1 = GaussBVP(sigmainitial = self.boundary.gauss_boundary["sigmainitial"],
                        sigmafinal = self.boundary.gauss_boundary["sigmafinal"],
                        epsilon = self.boundary.params["epsilon"],
                        Tf = self.boundary.params["Tf"],
                        M = self.boundary.params["weight"])
        
        if self.bridge_type == "half": #solve halfbridge
            sol1 = bvp1.get_half_sol()
        elif self.bridge_type=="marginal half": #solve for marginal half bridge
            sol1 = bvp1.get_marginal_half_sol()
        elif self.bridge_type=="full": #solve for full bridge
            sol1 = bvp1.get_full_sol()
        
        bvp1.save_csvs(sol1,self.folder_name) #save the csv 
    
    def get_ref_drift_gaussian(self,time):
        """Return linear coefficient of gaussian ref drift"""
        
        arr = self.get_underdamped_drift_array()
        v1 = self.boundary.params["epsilon"]*np.squeeze(arr[np.where(arr[:,0]==time),-1])/(2**0.5)

        return -v1
        #self.boundary.params["epsilon"]*np.squeeze(arr[np.where(arr[:,0]==time),-1])/(2**0.5)

    def get_underdamped_drift_array(self):
        """load csv of gaussian underdamped drift"""
        try:
            arr = np.loadtxt(open(self.underdamped_loc),
                        delimiter=",",skiprows=1) 
        except FileNotFoundError:
            print("Reference solution not found. Integrating First order system...")
            self.solve_bvp_gaussian()
            arr = np.loadtxt(open(self.underdamped_loc),
                        delimiter=",",skiprows=1) 
        return arr

    def get_underdamped_cumulants_array(self):
        """load csv of gaussian underdamped cumulants"""
        try:   
            arr = np.loadtxt(open(self.underdamped_cumulants_loc),
                        delimiter=",",skiprows=1) 
        
        except FileNotFoundError:
            print("Reference solution not found. Integrating First order system...")
            self.solve_bvp_gaussian()
            arr = np.loadtxt(open(self.underdamped_loc),
                        delimiter=",",skiprows=1) 
        
        return arr
        
    def get_underdamped_tgrid(self):
        arr = self.get_underdamped_cumulants_array()
        return np.squeeze(arr[:,0])

    def get_perturbative_cumulant(self,cumulant_type):
        arr = self.get_underdamped_cumulants_array() 

        if cumulant_type == "Position Variance": 
            temp = np.squeeze(arr[:,1])
        elif cumulant_type == "Momentum Variance":
            temp = np.squeeze(arr[:,3])
        elif cumulant_type == "Cross Correlation":
            temp = np.squeeze(arr[:,2])
        elif cumulant_type == "Momentum Mean":
            temp = np.zeros_like(np.squeeze(arr[:,0]))
        elif cumulant_type == "Position Mean":
            temp = np.zeros_like(np.squeeze(arr[:,0]))
        else:
            print("Perturbative prediction for "+cumulant_type+" not available.")
            temp = np.zeros_like(np.squeeze(arr[:,0]))
        
        return np.squeeze(arr[:,0]),temp
       
    def du_ref(self,t,q):
        """Return drift for t,q coordinate"""

        v1 = self.get_ref_drift_gaussian(t)
        try:
            return np.squeeze(q*v1) #+ v0
        except: #return zeros
            #TODO handle this error better or add msg
            return np.zeros_like(q) 
 
    def get_quantity(self,time,quantity,data=None,arr=None,time_type="t0"):
        """
        Get quantities to plot for the reference solution 
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
                if the dataset is already loaded in the script, this can be used passed to the function 
            time_type : str
                if "t0", rescale the time to t2. 
        returns:
            x : arr 
                array of shape (n,) of positions
            y : arr
                specified quantity evaluated at points x, with shape (n,)
        """

        ##fetch the reference data for halfbridge
        #if arr is None: #default overdamped
        #    arr = self.get_ref_data(data=data)  

        qtemp = np.linspace(-6,6,1000)

        if quantity =="value":
            return qtemp,np.zeros_like(qtemp)

        elif quantity =="marginal":
            return qtemp,np.zeros_like(qtemp)

        elif quantity =="underdamped drift":
            return qtemp,self.du_ref(time,qtemp)

        elif quantity =="underdamped marginal":
            #arr_ = self.get_underdamped_cumulants_array()
            
            t_vec,pos_var = self.get_perturbative_cumulant("Position Variance")
            #_,cc = self.get_perturbative_cumulant("Cross Correlation")
            #_,mom_var = self.get_perturbative_cumulant("Momentum Variance")
            
            return qtemp,np.exp(-(qtemp**2)/(2*pos_var[np.where(t_vec==time)]))/(np.sqrt(2*np.pi*pos_var[np.where(t_vec==time)]))
        
        #return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,idx]) #((0.2**2)*np.squeeze(arrtemp[:,idx]))+np.squeeze(arrtemp[:,2])

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

    fig.savefig(reference_.folder_name+f"/{data}_{quantity}.png",bbox_inches="tight")        

if __name__=="__main__":
    import matplotlib.pyplot as plt
    boundary = UnderdampedLandauer()
    reference_ = RefSolution(boundary)
    plot_data(reference_,quantity="value",data="overdamped")
    plot_data(reference_,quantity="marginal",data="overdamped")
    plot_data(reference_,quantity="underdamped drift",data="perturbative underdamped")
