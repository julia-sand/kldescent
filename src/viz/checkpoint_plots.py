import os
import yaml 

import pdb
from mpl_toolkits import mplot3d

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

        #if folder == None:
        #    self.folder = "tests/"+f"{boundary_cond.__class__.__name__}"
        #else: #if a folder path is provided, assume that the reference solution is there and overwrite the default    
        #    self.folder = folder

        self.folder = folder
        self.boundary_cond.overwrite_ref_path(self.folder)
            
        #make new folder for the results
        os.makedirs(self.folder+f"/plots_{self.epoch}", exist_ok = True) 
        self.plotter.setresultsdir(self.folder+f"/plots_{self.epoch}")
       
    def checkpoint_plots_du(self):
        """
        Runs the functions to make plots to show the results of the 
        drift learning
        """

        self.load_du_checkpoint()

        self.plotter.plot_drift(times = [0,self.boundary_cond.params["Tf"]/2,self.boundary_cond.params["Tf"]],net=self.net)
        self.make_sde_plots()

    def make_sde_plots(self):
            
        final_plot_time = self.boundary_cond.params["Tf"]
        
        t_vec, xinit, xref, cumulants_nn, cumulants_ref = self.integrator.compute_sde(self.net,final_plot_time)
        
        self.plotter.update_final_cumulants(cumulants_nn[-1,:])
        self.plotter.marginal_distribution([xinit,xref],final_plot_time)
        self.plotter.plot_cumulants(t_vec,cumulants_nn,cumulants_ref,self.data_loaded["testsamples"])
        self.plotter.save_cumulants_csv(t_vec,cumulants_nn)

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
        #self.plot_pdf_3d()
    
    def compute_rho_nn(self,x_samples=None):
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

        with torch.no_grad():
        
            if x_samples==None:
                self.data_loaded["testsamples"]
            #generate xmesh
            Q,P = generate_mesh(x_samples,qmax=6)
            xinit = torch.tensor(np.column_stack((Q.flatten(),P.flatten())),dtype=torch.float32,requires_grad=False)
            
            #torch.tensor(self.boundary_cond.generate_samples(x_samples,"initial"),dtype=torch.float32,requires_grad=False)
            pdf_out = self.rhonet(xinit.detach().clone())
        
            return xinit.detach().numpy(),pdf_out.detach().numpy()


    def compute_rho_girsanov(self,x_samples=None,mc_samples=200):
        """
        Convenience function that returns an estimate of the final joint distribution using girsanov theorem
        Used for testing

        args:
            x_samples:
                number of space discretisation steps
        returns:
            xout: 
                2d mesh of points of the statespace
            jointout:
                estimated joint distribution using girsanov theorem
        """
        
        #initialise drift network from checkpoint and set up integrator
        self.load_du_checkpoint()
        fpk_integrator = self.load_fpk_integrator(self.net)

        #default number of samples from parameter file
        if x_samples==None:
            x_samples = self.data_loaded["testsamples"]
        
        #generate xmesh
        Q,P = generate_mesh(x_samples,qmax=6)
        xinit = torch.tensor(np.column_stack((Q.flatten(),P.flatten())),dtype=torch.float32,requires_grad=False)
        
        #torch.tensor(self.boundary_cond.generate_samples(x_samples,"initial"),dtype=torch.float32,requires_grad=False)
        x2 = xinit.detach().clone() 

        pdf_out = torch.zeros_like(x2[...,[0]],requires_grad=False,dtype=torch.float32)
        #0*xinit[...,[0]]

        #want shape = (x,2)
        with torch.no_grad():
            for _ in range(mc_samples):
                #temp = self.fpk_integrator.compute_pdf(x2.detach().clone())
                pdf_out += self.fpk_integrator.compute_pdf(x2.detach().clone())
                #print(torch.max(temp))
        
        return xinit.detach().numpy(),pdf_out.detach().numpy()/mc_samples

    def make_contour_plot2(self):
        xout,jointout = self.compute_rho_girsanov(x_samples=20)
        fig,ax = plt.subplots()
        ax, ax_pmarginal,ax_qmarginal = self.plotter.scatter_panel_layout(ax)

        self.plotter.joint_distributions_scatter_panel(ax,
                                                    ax_pmarginal,
                                                    ax_qmarginal,
                                                    jointout,
                                                    xout,
                                                    time=None,vmax=0.5)
        fig.savefig(self.folder+f"/plots_{self.epoch}"+"/test8_girs.png",bbox_inches="tight")        

        plt.close()

    def plot_pdf_3d(self):
        xs,jointout = self.compute_rho_nn(x_samples=50)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        z = jointout.flatten()
        x = xs[...,0]
        y = xs[...,1]

        ax.plot3D(x, y, z, 'green')
        ax.set_title('3D Line Plot')
        plt.savefig(self.folder+f"/plots_{self.epoch}"+"/test_3d.png",bbox_inches="tight")        


    def make_contour_plot(self):
        xout,jointout = self.compute_rho_nn(x_samples=20)
        xout2,jointout2 = self.compute_rho_girsanov(x_samples=20)
        
        fig,ax = plt.subplots()
        ax, ax_pmarginal,ax_qmarginal = self.plotter.scatter_panel_layout(ax)

        self.plotter.joint_distributions_scatter_panel(ax,
                                                    ax_pmarginal,
                                                    ax_qmarginal,
                                                    jointout,
                                                    xout,
                                                    time=None,vmax=0.5)
        self.plotter.joint_distributions_scatter_panel(ax,
                                                    ax_pmarginal,
                                                    ax_qmarginal,
                                                    jointout2,
                                                    xout2,
                                                    time=None,vmax=0.5)
        fig.savefig(self.folder+f"/plots_{self.epoch}"+"/test8_nn.png",bbox_inches="tight")        

        plt.close()

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