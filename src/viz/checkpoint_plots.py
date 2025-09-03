import os
import yaml 

import pdb

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
    1. init neural networks
    1a. load checkpoint based on input folder and epoch
    2. initialises plotter based on boundary conditions either from cmd or parameter file
    3. sets the location of the reference solution based on input folder
    4. creates new plots folder in the folder location
    5. initialises the integrator to use for the the plots
    """

    def __init__(self,folder=None,epoch=None):
        
        #get folder and load boundary data and params from file
        if folder is None:
            folder = fetch_results_folder_from_cmd()
        else:
            folder = folder 

        boundary_cond,data_loaded = init_params_from_file(folder)

        params = checkpoint_plot_params() #get checkpoint plot parameters from cmd line
        
        self.boundary_cond = boundary_cond
        self.data_loaded = data_loaded
        
        if epoch is None:
            self.epoch = params["epoch"]
        else:
            self.epoch=epoch
        
        self.plotter = PlotFuns(boundary_cond)

        if folder == None:
            self.folder = "tests/"+f"{boundary_cond.__class__.__name__}"
        else: #if a folder path is provided, assume that the reference solution is there and overwrite the default    
            self.folder = folder
            boundary_cond.overwrite_ref_path(self.folder)
            

        #make new folder for the results
        os.makedirs(self.folder+f"/plots_{self.epoch}", exist_ok = True) 
        self.plotter.setresultsdir(self.folder+f"/plots_{self.epoch}")
       
    def checkpoint_plots_du(self):
        """
        Runs the functions to make plots to show the results of the 
        drift learning
        """

        self.load_du_checkpoint()

        self.integrator.compute_sde(self.net,
                        self.plotter,
                        self.boundary_cond.params["Tf"])
        
        self.plotter.plot_drift(times = [0,self.boundary_cond.params["Tf"]/2,self.boundary_cond.params["Tf"]],net=self.net)

    def load_du_checkpoint(self):
        self.net = NeuralNetwork(self.boundary_cond,self.data_loaded)
        self.net.load_checkpoint(self.folder+f"/checkpoints/{self.net.net_name}_checkpoint_{self.epoch}.pt")
        self.integrator = langevin.SDE(self.boundary_cond,sde_params=self.data_loaded)

    def load_rho_checkpoint(self):
        self.rhonet = RhoNeuralNetwork(self.data_loaded)
        self.rhonet.load_checkpoint(self.folder+f"/checkpoints/{self.rhonet.net_name}_checkpoint_{self.epoch}.pt")
        print(self.folder+f"/checkpoints/{self.rhonet.net_name}_checkpoint_{self.epoch}.pt")

    def load_fpk_integrator(self,net):
        self.fpk_integrator = fpk.FokkerPlanck(self.boundary_cond,net,
                                                sde_params=self.data_loaded)
              
    def checkpoint_plots_rho(self):
        self.load_rho_checkpoint()
        self.make_contour_plot()
        #self.make_contour_plot2()
        
    def compute_rho_nn(self,x_samples=100):
        """
        Generates a statespace grid and evaluates the neural network estimate of the 
        joint probability distribution at final time 

        args:
            x_samples:
                number of space discretisation steps
        returns:
            xout,jointout
            -2d mesh of points of the statespace
            -evaluated neural network joint distribution
        """
        
        #generate xmesh
        Q,P = generate_mesh(x_samples,qmax=6)
        xinit = torch.tensor(np.column_stack((Q.flatten(),P.flatten())),dtype=torch.float32,requires_grad=False)
        
        #want shape = (x,2)
        with torch.no_grad():
            pdf_out = self.rhonet(xinit.detach().clone())
        
        #pdb.set_trace()

        return xinit.detach().numpy(),pdf_out.detach().numpy()


    def compute_rho_girsanov(self,x_samples=30,mc_samples=400):
        """
        Convenience function that returns an estimate of the final joint distribution using girsanov theorem
        Used for testing

        args:
            x_samples:
                number of space discretisation steps
        returns:
            xout,jointout
            -2d mesh of points of the statespace
            -estimated joint distribution using girsanov theorem
        """
        
        self.load_du_checkpoint()
        fpk_integrator = self.load_fpk_integrator(self.net)

        #generate xmesh
        Q,P = generate_mesh(x_samples,qmax=4)
        xinit = torch.tensor(np.column_stack((Q.flatten(),P.flatten())),dtype=torch.float32,requires_grad=False)
        x2 = xinit.detach().clone() 
        pdf_out = 0*xinit[...,[0]]
        #want shape = (x,2)
        with torch.no_grad():
            for _ in range(mc_samples):
                #temp = self.fpk_integrator.compute_pdf(x2.detach().clone())
                pdf_out += self.fpk_integrator.compute_pdf(x2.detach().clone())
                #print(torch.max(temp))
        return xinit.detach().numpy(),pdf_out.detach().numpy()/mc_samples

    def make_contour_plot2(self):
        xout,jointout = self.compute_rho_girsanov(x_samples=50)
        fig,ax = plt.subplots()

        self.plotter.joint_distributions_scatter_panel(ax,
                                                    jointout,
                                                    xout,
                                                    time=None,vmax=0.5)
        fig.savefig(self.folder+f"/plots_{self.epoch}"+"/test8_girs.png",bbox_inches="tight")        

    def make_contour_plot(self):
        xout,jointout = self.compute_rho_nn(x_samples=50)
        fig,ax = plt.subplots()

        self.plotter.joint_distributions_scatter_panel(ax,
                                                    jointout,
                                                    xout,
                                                    time=None,vmax=0.5)
        fig.savefig(self.folder+f"/plots_{self.epoch}"+"/test8_nn.png",bbox_inches="tight")        

def compute_ref_kl(boundary):
    times = boundary.get_reference_timegrid()
    klout = np.empty(len(times))
    #mean_t = boundary.get_known_solution_cumulants("Position Mean")
    var_t = boundary.get_known_solution_cumulants("Position Variance")
    for t in enumerate(times):
        xref,duref = boundary.get_reference_data(t[1])
        ptx = plotter.normal_distribution(xref,0,var_t[t[0]])
        klout[t[0]] = np.trapz(duref*ptx,xref)

    return np.trapz(klout,times)

if __name__=="__main__":

    plot_utils = PlotUtils()

    plot_utils.checkpoint_plots_du()
    plot_utils.checkpoint_plots_rho()