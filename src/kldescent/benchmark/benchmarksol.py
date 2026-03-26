import os

import numpy as np 

from kldescent.sde.boundary import *
from kldescent.utils.parser import fetch_results_folder_from_cmd
from kldescent.model.nn import MLP,NeuralNetwork
from kldescent.benchmark.utilfunctions import *
from kldescent.benchmark.perturbative.driftinterp import perturbative_du_interp,train_and_save_perturbative_du
from kldescent.benchmark.perturbative.corrections import MarginalHalfBridgePerturbation,HalfBridgePerturbation
from kldescent.benchmark.overdamped.sol import compute_overdamped
from .gaussian import GaussBVP

class BaseBenchmarkSolution():

    def __init__(self, bridge_type="half", boundary=None, folder_name=None):
        
        self.boundary = boundary

        if folder_name is not None: #use this if running the file directly
            self.folder_name = folder_name
        else:
            self.folder_name = fetch_results_folder_from_cmd() #f"{self.boundary.__class__.__name__}"
                
        os.makedirs(self.folder_name,exist_ok=True)

        #sets benchmark file locations
        self.overdamped_loc = self.folder_name+"/overdamped.csv"
        self.perturb_underdamped_loc = self.folder_name+"/perturbative_underdamped.csv"
        self.ref_loc = self.folder_name+"/benchmark.csv"
        self.perturbative_cumulants_loc = self.folder_name+"/perturbative_cumulants.csv"

    def get_perturbative_calculations(self,bridge_type,overdamped_arr):
        """initialise the class to compute the perturbative solution 
        depending on the bridge type
        args: 
            bridge_type : str
                "marginal", "half" 
        """

        if bridge_type == "marginal":
            pertsol = MarginalHalfBridgePerturbation(self.boundary,overdamped_arr,self.folder_name)
        
        elif bridge_type == "half":  #update this to compute the half bridge solution
            pertsol = HalfBridgePerturbation(self.boundary,overdamped_arr,self.folder_name)
            
        else: 
        
            pertsol = HalfBridgePerturbation(self.boundary,overdamped_arr,self.folder_name)

        return pertsol

class PerturbativeBenchmarkSolution(BaseBenchmarkSolution):
    r"""
    Class used for organising the location
    for the benchmark solutions computed using perturbation theory
    
    Methods:
        initialise_net()
        get_benchmark_path()
        get_timegrid()
            Fetches time discretisation from benchmark solution
        get_benchmark_data()
            fetches benchmark data found in benchmark sol path
        get_quantity()
        get_perturbative_cumulants_array()
        get_perturbative_cumulant()
    """

    def __init__(self, bridge_type="half", boundary=None, folder_name=None):
        r"""
        Initialises parameters and directs to the benchmark solutions. 
        If no benchmark solution is available in the folder, then computes the benchmark solution
        args:
            bridge_type : str
                type of bridge. Choose from "marginal" or "half". 
            boundary : class
                specifies the boundary conditions
            folder_name : str
                if no boundary condition specified, the folder can be set manually using this parameter. 
        """
        
        BaseBenchmarkSolution.__init__(self,bridge_type,boundary,folder_name)
        self.initialise_net(bridge_type)

    def initialise_net(self,bridge_type):

        #initialise a simple neural net to approximate the drift 
        self.du_net_benchmark = NeuralNetwork(self.boundary) #MLP()

        #get perturbative sol 
        try:
            #compute corrections
            perturbative_du_interp(self,self.du_net_benchmark)
        
        except FileNotFoundError: #no csvs available to train from

            try: 
                #try to fetch overdamped results
                overdamped_arr = self.get_benchmark_data(data="overdamped")

            except FileNotFoundError:
                print("Overdamped results not available. Solving cell problem...")

                #compute overdamped solution
                compute_overdamped(self.boundary,
                                    self.get_benchmark_path(),
                                    bridge_type=bridge_type)

                print("Cell problem completed.")
                overdamped_arr = self.get_benchmark_data(data="overdamped")

            print("Perturbative results not available. Computing corrections...")

            #initialise correct bridge type
            pertsol = self.get_perturbative_calculations(bridge_type,overdamped_arr)
            pertsol.save_perturbative()
            pertsol.save_perturbative_cumulants()

            #initialise correct bridge type
            pertsol = self.get_perturbative_calculations(bridge_type,overdamped_arr)
            
            pertsol.save_perturbative()
            pertsol.save_perturbative_cumulants()
            print("Perturbative Results saved.")
                
            train_and_save_perturbative_du(self,self.du_net_benchmark)
    
        self.du_net_benchmark.eval()

    def get_benchmark_path(self):
        """return the folder where the benchmark solution files are"""

        return self.folder_name # os.path.dirname(__file__) + "/" + self.folder_name
    
    def get_benchmark_data(self,data=None):
        """Return an array for benchmark solution"""
        
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

    def get_timegrid(self,time=None):
        arr = self.get_benchmark_data(data="overdamped")
        return np.unique(arr[:,0])
        
    def get_quantity(self,time,quantity,data=None,arr=None,time_type="t2"):
        """
        Get quantities to plot for the benchmark solution 
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

        if arr is None: #default overdamped
            arr = self.get_benchmark_data(data="overdamped")  

        if time_type=="t0":
            #multiply by eps**2 to get t2 from t0 
            time *= self.boundary.params["epsilon"]**2

        ##fetch the approximation data
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

    def get_perturbative_cumulants_array(self,idx=None):
        """return the array for predicted cumulants using perturbation theory"""
        arr = np.loadtxt(open(self.perturbative_cumulants_loc),
                        delimiter=",",skiprows=1,usecols=idx) 
        return arr 

    def get_perturbative_cumulant(self,cumulant_type):
        #arr = self.get_perturbative_cumulants_array() 
        try:  
            if cumulant_type == "Position Variance": 
                #temp = np.squeeze(arr[:,3])
                arr = self.get_perturbative_cumulants_array(idx= (0,3))
            elif cumulant_type == "Momentum Variance":
                #temp = np.squeeze(arr[:,4])
                arr = self.get_perturbative_cumulants_array(idx= (0,4))

            elif cumulant_type == "Cross Correlation":
                #temp = np.squeeze(arr[:,5])
                arr = self.get_perturbative_cumulants_array(idx= (0,5))

            elif cumulant_type == "Momentum Mean":
                #temp = np.squeeze(arr[:,2])
                arr = self.get_perturbative_cumulants_array(idx= (0,2))

            elif cumulant_type == "Position Mean":
                #temp = np.squeeze(arr[:,1])
                arr = self.get_perturbative_cumulants_array(idx= (0,1))

            else:
                print("Perturbative prediction for "+cumulant_type+" not available.")
                #temp = np.zeros_like(np.squeeze(arr[:,0]))
                arr = np.zeros((10,2))
        except IndexError:
            print("Perturbative prediction for "+cumulant_type+" not available.")
            return np.zeros(10),np.zeros(10)

        return np.squeeze(arr[:,0]),np.squeeze(arr[:,1])


class GaussBenchmarkSolution(BaseBenchmarkSolution):
    r"""Compile the gaussian versions of the benchmark solutions
    Benchmark solution is calculated as the solution of the first order 
    differential equations

    Methods
        solve_bvp_gaussian()
        get_underdamped_drift_array()
        get_drift_gaussian()
        get_underdamped_cumulants_array()
        get_timegrid()
        get_perturbative_cumulant()
        du_benchmark()
        get_quantity()
    """

    def __init__(self,bridge_type="half",boundary=None,folder_name=None):   
        
        BaseBenchmarkSolution.__init__(self,bridge_type="half",boundary=boundary,folder_name=folder_name)
        
        
        #set different filename for underdamped cumulants 
        self.underdamped_cumulants_loc = self.folder_name+"/underdamped_cumulants.csv"
        self.underdamped_loc = self.folder_name+"/underdamped.csv"
        self.du_net_benchmark = None

        #set the bridge type
        self.bridge_type = bridge_type

    def solve_bvp_gaussian(self,):

        #solve BVP for gaussian 
        bvp1 = GaussBVP(sigmainitial = self.boundary.gauss_boundary["sigmainitial"],
                        sigmafinal = self.boundary.gauss_boundary["sigmafinal"],
                        epsilon = self.boundary.params["epsilon"],
                        Tf = self.boundary.params["Tf"],
                        M = self.boundary.params["weight"])
        
        if self.bridge_type == "half": #solve halfbridge
            sol1 = bvp1.get_half_sol()
        elif self.bridge_type=="marginal": #solve for marginal half bridge
            sol1 = bvp1.get_marginal_half_sol()
        elif self.bridge_type=="full": #solve for full bridge
            sol1 = bvp1.get_full_sol()
        
        bvp1.save_csvs(sol1,self.folder_name) #save the csv 
    
    def get_underdamped_drift_array(self):
        """load csv of gaussian underdamped drift"""
        try:
            arr = np.loadtxt(open(self.underdamped_loc),
                        delimiter=",",skiprows=1,usecols=(0,-1)) 
        except FileNotFoundError:
            print("Benchmark solution not found. Integrating First order system...")
            self.solve_bvp_gaussian()
            arr = np.loadtxt(open(self.underdamped_loc),
                        delimiter=",",skiprows=1,usecols=(0,-1)) 
        return arr

    def get_drift_gaussian(self,time):
        """Return linear coefficient of gaussian ref drift"""
        
        arr = self.get_underdamped_drift_array()
        v1 = self.boundary.params["epsilon"]*np.squeeze(arr[np.where(arr[:,0]==time),-1])/(2**0.5)

        return -v1
        #self.boundary.params["epsilon"]*np.squeeze(arr[np.where(arr[:,0]==time),-1])/(2**0.5)


    def get_underdamped_cumulants_array(self,idx=None):
        """load csv of gaussian underdamped cumulants
        args: 
            idx : tuple
                indices of the columns to load. 
        """
        try:   
            arr = np.loadtxt(open(self.underdamped_cumulants_loc),
                        delimiter=",",skiprows=1,usecols=idx) 
        
        except FileNotFoundError:
            print("Benchmark solution not found. Integrating First order system...")
            self.solve_bvp_gaussian()
            arr = np.loadtxt(open(self.underdamped_loc),
                        delimiter=",",skiprows=1,usecols=idx) 
        
        return arr
        
    def get_timegrid(self):
        arr = self.get_underdamped_cumulants_array(idx = 0)
        return np.squeeze(arr)

    def get_perturbative_cumulant(self,cumulant_type):
        """Fetch saved cumulants for plotting
        args 
            cumulant_type : str
                Name of the cumulant 
        returns 
            tvec : (n,) array
                vector of times
            cumulantvec : (n,) array
                vector of cumulant values
        """

        if cumulant_type == "Position Variance": 
            arr = self.get_underdamped_cumulants_array(idx=(0,1)) 
        elif cumulant_type == "Momentum Variance":
            arr = self.get_underdamped_cumulants_array(idx=(0,3))
        elif cumulant_type == "Cross Correlation":
            arr = self.get_underdamped_cumulants_array(idx=(0,2)) 
        elif cumulant_type == "Momentum Mean":
            arr = np.zeros((10,2))
            #temp = np.zeros_like(np.squeeze(arr[:,0]))
        elif cumulant_type == "Position Mean":
            arr = np.zeros((10,2))
            #temp = np.zeros_like(np.squeeze(arr[:,0]))
        else:
            print("Perturbative prediction for "+cumulant_type+" not available.")
            arr = np.zeros((10,2))
        
        return np.squeeze(arr[:,0]),np.squeeze(arr[:,1])
       
    def du_benchmark(self,t,q):
        """Return drift for t,q coordinate"""

        v1 = self.get_drift_gaussian(t)
        try:
            return np.squeeze(q*v1) #+ v0
        except: #return zeros
            #TODO handle this error better or add msg
            return np.zeros_like(q) 
 
    def get_quantity(self,time,quantity,data=None,arr=None,time_type="t0"):
        """
        Get quantities to plot for the benchmark solution 
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

        qtemp = np.linspace(-6,6,1000)

        if quantity =="value":
            return qtemp,np.zeros_like(qtemp)

        elif quantity =="marginal":
            return qtemp,np.zeros_like(qtemp)

        elif quantity =="underdamped drift":
            return qtemp,self.du_benchmark(time,qtemp)

        elif quantity =="underdamped marginal":
            #arr_ = self.get_underdamped_cumulants_array()
            
            t_vec,pos_var = self.get_perturbative_cumulant("Position Variance")
            #_,cc = self.get_perturbative_cumulant("Cross Correlation")
            #_,mom_var = self.get_perturbative_cumulant("Momentum Variance")
            
            return qtemp,np.exp(-(qtemp**2)/(2*pos_var[np.where(t_vec==time)]))/(np.sqrt(2*np.pi*pos_var[np.where(t_vec==time)]))
        
        #return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,idx]) #((0.2**2)*np.squeeze(arrtemp[:,idx]))+np.squeeze(arrtemp[:,2])

def plot_data(benchmark_,quantity,data):
    
    times = [0,0.06,0.12,0.2]

    fig,axs = plt.subplots(ncols=len(times),nrows=1)
    fig.set_figheight(15)
    fig.set_figwidth(25)

    for time in enumerate(times):
        q, ptq = benchmark_.get_quantity(time[1], quantity, data=data)
        axs[time[0]].plot(q,ptq)
        axs[time[0]].set_title(f"t={time[1]}")
        
    fig.tight_layout()

    fig.savefig(benchmark_.folder_name+f"/{data}_{quantity}.png",bbox_inches="tight")        

if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    boundary = UnderdampedLandauer()
    benchmark_ = PerturbativeBenchmarkSolution(boundary)
    plot_data(benchmark_,quantity="value",data="overdamped")
    plot_data(benchmark_,quantity="marginal",data="overdamped")
    plot_data(benchmark_,quantity="underdamped drift",data="perturbative underdamped")
