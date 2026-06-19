import os

import numpy as np 

#from kldescent.sde.boundary import *
from kldescent.utils.parser import fetch_results_folder_from_cmd
from kldescent.model.nn import NeuralNetwork
from kldescent.benchmark.utilfunctions import *
from kldescent.benchmark.perturbative.driftinterp import *
from kldescent.benchmark.perturbative.corrections import MarginalHalfBridgePerturbation,HalfBridgePerturbation, FullBridgePerturbation
from kldescent.benchmark.overdamped.computesol import compute_overdamped
from kldescent.benchmark.gaussian import GaussBVP

class BaseBenchmarkSolution():

    def __init__(self, bridge_type, boundary=None, folder_name=None):
        
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
        """
        Initialise and return objects required for perturbative calculations
        based on the selected bridge type.

        Parameters
        ----------
        bridge_type : str
            Type of bridge to use. Must be "marginal", "half" or "full".
        overdamped_arr : np.ndarray
            Precomputed overdamped solution data used as input for the
            perturbative calculations.

        Returns
        -------
        object
            Instance configured to compute the perturbative solution for the specified bridge type.
        """

        if bridge_type == "marginal":
            pertsol = MarginalHalfBridgePerturbation(self.boundary.params,overdamped_arr,self.folder_name)

        elif bridge_type == "half":
            pertsol = HalfBridgePerturbation(self.boundary.params,overdamped_arr,self.folder_name)

        elif bridge_type == "full":
            pertsol = FullBridgePerturbation(self.boundary.params,overdamped_arr,self.folder_name)

        else:
            raise ValueError(f"Unknown bridge_type {bridge_type!r}. Must be 'marginal', 'half', or 'full'.")

        return pertsol

class PerturbativeBenchmarkSolution(BaseBenchmarkSolution):
    r"""
    Utility class for managing paths and data access for benchmark solutions
    computed using perturbation theory.

    This class centralizes file handling and provides convenient access to
    time grids, benchmark datasets, and perturbative quantities.

    Methods:
        initialise_net():
            Initialize the underlying network or model used for computations.

        get_benchmark_path():
            Return the filesystem path where benchmark solutions are stored.

        get_timegrid():
            Retrieve the time discretization associated with the benchmark data.

        get_benchmark_data():
            Load and return the benchmark dataset from the configured path.

        get_benchmark_quantity():
            Extract a specific quantity from the benchmark dataset.

        get_perturbative_cumulants_array():
            Return all perturbative cumulants as an array.

        get_perturbative_cumulant():
            Retrieve a single perturbative cumulant.
    """

    def __init__(self, bridge_type="half", boundary=None, folder_name=None):
        r"""
        Initialise parameters and configure access to benchmark solutions.

        If a benchmark solution is not found in the specified location, it will
        be computed and stored automatically.

        Parameters
        ----------
        bridge_type : str
            Type of bridge to use. Must be either "marginal" or "half".
        boundary : object, optional
            Object specifying the boundary conditions.
        folder_name : str, optional
            Manually specify the folder containing benchmark solutions.
        """
        
        self.bridge_type = bridge_type
        BaseBenchmarkSolution.__init__(self,self.bridge_type,boundary,folder_name)
        #self.initialise_net()
        self.du_net_benchmark = None

    def initialise_net(self,bridge_type,nn_params=None):

        #initialise a simple neural net to approximate the drift 
        self.du_net_benchmark = NeuralNetwork(self.boundary,nn_params=nn_params) #MLP()

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
            print("Perturbative Results saved.")
                
            train_and_save_perturbative_du(self,self.du_net_benchmark)
    
        self.du_net_benchmark.eval()

    def get_benchmark_path(self):
        """return the folder where the benchmark solution files are"""

        return self.folder_name # os.path.dirname(__file__) + "/" + self.folder_name
    
    def get_benchmark_data(self,data=None):
        """Return an array for benchmark solution"""
        #try:    
        return self.get_arr(data)
        #except FileNotFoundError:
        #    self.initialise_net(self,bridge_type)
        #    return self.get_arr(data)

    def get_arr(self,data=None):
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
        """Return the t0 time grid"""
        arr = self.get_benchmark_data(data="perturbative underdamped")
        return np.unique(arr[:,0])
        
    def get_benchmark_quantity(self,time,quantity,data,arr=None,time_type="t0"):
        """
        Retrieve a quantity from the benchmark solution for plotting.

        Parameters
        ----------
        time : float
            Time at which to evaluate the data.
        quantity : str
            Quantity to extract from the dataset. Options are:
                - "value": overdamped value function
                - "marginal": overdamped probability distribution
                - "underdamped drift": perturbative estimate of the underdamped drift
                - "underdamped marginal": perturbative marginal distribution of position
        data : str
            Dataset to use. Must be one of:
                - "overdamped": cell problem solution
                - "underdamped": saved perturbative solution
        arr : np.ndarray, optional
            Preloaded dataset. If provided, this is used instead of loading
            data from disk.
        time_type : str, optional
            If set to "t0", the time is rescaled to t2 by dividing by epsilon squared.

        Returns
        -------
        x : np.ndarray
            One-dimensional array of positions with shape (n,).
        y : np.ndarray
            Array of the selected quantity evaluated at `x`, shape (n,).
        """

        #ensure correct time scale
        if data == "overdamped":
            if time_type=="t0":
                #multiply by eps**2 to get t2 from t0 
                time *= self.boundary.params["epsilon"]**2
        if data == "perturbative underdamped":
            if time_type=="t2":
                #divide by eps**2 to get t0 from t2 
                time /= self.boundary.params["epsilon"]**2
               
        if quantity =="value":
            
            index2=3
             
        elif quantity =="marginal":
            index2=2
        
        elif quantity =="underdamped drift":
            index2=-1
        
        elif quantity =="underdamped marginal":
            index2=-2
        
        if arr is None: 
            arr = self.get_benchmark_data(data=data) 
        
        ##fetch the approximation data
        arrtemp = arr[np.where(arr[:,0]==np.round(time,6))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x
    
        return np.squeeze(arrtemp[:,1]),np.squeeze(arrtemp[:,index2]) 

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

    def du_benchmark(self,time,q):
        """Return drift for t,q coordinate
        Warning: only use this function for plotting if there is no other option. 
        
        THIS WILL RUN ****VERY**** SLOWLY FOR INTEGRATION
        time should be given as t0
        """
        
        arr = self.get_benchmark_data(data="perturbative underdamped")
        arrtemp = arr[np.where(arr[:,0]==np.round(time,6))] 
        arrtemp = arrtemp[arrtemp[:,1].argsort()]  #sort by x
        x = arrtemp[:,1].flatten()
        y = arrtemp[:,-1].flatten()
        du_out = np.interp(q,x,y)
        return -(self.boundary.params["epsilon"])*du_out.flatten()/(2**0.5)
 

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
        get_benchmark_quantity()
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
        """
        Load CSV of Gaussian underdamped cumulants.

        Parameters
        ----------
        idx : tuple, optional
            Indices of the columns to load.
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
        """
        Fetch saved cumulants for plotting.

        Parameters
        ----------
        cumulant_type : str
            Name of the cumulant.

        Returns
        -------
        tvec : np.ndarray, shape (n,)
            Vector of times.
        cumulantvec : np.ndarray, shape (n,)
            Vector of cumulant values.
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
                
        if v1 is None: 
            return np.zeros_like(q)

        return v1*q#np.squeeze(q*v1) 
        
    def get_benchmark_quantity(self,time,quantity,data=None,arr=None,time_type="t0"):
        """
        Get quantities to plot for the benchmark solution.

        Parameters
        ----------
        time : float
            The time at which to look up the data.
        quantity : str
            Quantity to return from the data table. Choices are: "value"
            (overdamped value function), "marginal" (overdamped distribution),
            "underdamped drift" (perturbative estimate of underdamped DU),
            and "underdamped marginal" (perturbative underdamped marginal of
            the position).
        data : str, optional
            Dataset to use. Choose from "overdamped" (cell problem solution)
            or "underdamped" (saved perturbative solution).
        arr : np.ndarray, optional
            If the dataset is already loaded in the script, it can be passed
            to the function.
        time_type : str, optional
            If "t0", rescale the time to t2.

        Returns
        -------
        x : np.ndarray, shape (n,)
            Array of positions.
        y : np.ndarray, shape (n,)
            Specified quantity evaluated at points x.
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

if __name__=="__main__":
    from kldescent.sde.boundary import *
    from kldescent.utils.parser import fetch_results_folder_from_cmd
    from kldescent.utils.parser import overdamped_argparser

    bridgetype = overdamped_argparser()["bridgetype"]

    boundary = UnderdampedLandauer()
    benchmark_ = PerturbativeBenchmarkSolution(bridge_type=bridgetype,boundary=boundary,
                        folder_name=fetch_results_folder_from_cmd())
    benchmark_.initialise_net(benchmark_.bridge_type)
    #plot_data(benchmark_,quantity="value",data="overdamped")
    #plot_data(benchmark_,quantity="marginal",data="overdamped")
    #plot_data(benchmark_,quantity="marginal",data="perturbative underdamped")
    #plot_data(benchmark_,quantity="underdamped drift",data="perturbative underdamped")
    #plot_benchmark_drift(benchmark_)
