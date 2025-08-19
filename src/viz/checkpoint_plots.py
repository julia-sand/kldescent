import os
import yaml 

import torch
import numpy as np

from ..model.nn import RhoNeuralNetwork,NeuralNetwork
from .plots import PlotFuns
from ..sde import langevin,fpk
from ..utils.parser import checkpoint_plot_params,fetch_results_folder_from_cmd
from ..utils.getboundary import *
from .contour import *

class PlotUtils():
    """
    Class which prepares the neural net, boundary and plotting variables to
    set up the plots
    Functionality:
    1. load neural network checkpoint based on input folder and epoch
    2. initialises plotter based on boundary conditions
    3. sets the location of the reference solution based on input folder
    4. creates new plots folder in the folder location
    5. initialises the integrator to use for the the plots
    """

    def __init__(self):
        
        #get folder and load boundary data and params from file
        folder = fetch_results_folder_from_cmd()   
        boundary_cond,data_loaded = init_params_from_file(folder)

        params = checkpoint_plot_params() #get checkpoint plot parameters from cmd line
        
        self.boundary_cond = boundary_cond
        self.epoch = params["epoch"]
        self.net = NeuralNetwork(boundary_cond,data_loaded)
        self.rhonet = RhoNeuralNetwork(boundary_cond,data_loaded)
        
        self.plotter = PlotFuns(boundary_cond)

        if folder == None:
            self.folder = "tests/"+f"{boundary_cond.__class__.__name__}"
        else: #if a folder path is provided, assume that the reference solution is there and overwrite the default    
            self.folder = folder
            boundary_cond.overwrite_ref_path(self.folder)
            self.net.load_checkpoint(self.folder+f"/checkpoints/{self.net.net_name}_checkpoint_{self.epoch}.pt")
            self.rhonet.load_checkpoint(self.folder+f"/checkpoints/{self.rhonet.net_name}_checkpoint_{self.epoch}.pt")

        #make new folder for the results
        os.makedirs(self.folder+f"/plots_{self.epoch}", exist_ok = True) 
        self.plotter.setresultsdir(self.folder+f"/plots_{self.epoch}")

        self.integrator = langevin.SDE(self.boundary_cond,sde_params=data_loaded)
        self.fpk_integrator = fpk.FokkerPlanck(self.boundary_cond,self.net,
                                                sde_params=data_loaded)
        
    def checkpoint_plots_du(self):

        #folder = logger.get_results_path()
        self.integrator.compute_sde(self.net,
                        self.plotter,
                        self.boundary_cond.params["Tf"])
        self.plotter.plot_drift(times = [0,self.boundary_cond.params["Tf"]/2,self.boundary_cond.params["Tf"]],net=self.net)

    def checkpoint_plots_rho(self):
        """
        Specifies plotting functions to run to display results of training the rho network
        """

        #generate xmesh
        xinit = torch.tensor(generate_mesh(),requires_grad=False)
        pdf_out = self.integrator.compute_pdf(xinit)



def compute_ref_kl(boundary):
    times = boundary.get_reference_timegrid()
    klout = np.empty(len(times))
    #mean_t = boundary.get_known_solution_cumulants("Position Mean")
    var_t = boundary.get_known_solution_cumulants("Position Variance")
    for t in enumerate(times):
        xref,duref = boundary.get_reference_data(t[1])
        ptx = plotter.normal_distribution(xref,0,var_t[t[0]])
        #np.exp(-(xref-mean_t[t[0]])**2/(2*var_t[t[0]]))/(np.sqrt(2*np.pi*var_t[t[0]]))
        klout[t[0]] = np.trapz(duref*ptx,xref)
    return np.trapz(klout,times)

if __name__=="__main__":

    plot_utils = PlotUtils()

    #plot_utils.checkpoint_plots_du()
    plot_utils.checkpoint_plots_rho()