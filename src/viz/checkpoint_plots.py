import os
import yaml 

from mpl_toolkits import mplot3d

import torch
import numpy as np

from ..model.nn import RhoNeuralNetwork,NeuralNetwork
from .plots import PlotFuns,update_mpl
from ..sde import langevin
from ..utils.parser import checkpoint_plot_params,fetch_results_folder_from_cmd
from ..utils.getboundary import *
from .contour import *
from ..refsolution.reference import RefSolution,GaussRefSolution

def compute_rho_girsanov(integrator,x_samples=40,mc_samples=64,Tf=None):
    """
    Computes an estimate of the final joint distribution using girsanov theorem
    under the input drift 
    
    args:
        dunet : class 
            neural network approximating the drift
        integrator : class
            determines the method of SDE integration
        x_samples : int 
            number of space discretisation steps
        mc_samples : int
            number of MC samples. 
        Tf : float
            time for the plot
    returns:
        xout: 
            2d mesh of points of the statespace
        jointout:
            estimated joint distribution using girsanov theorem
    """
    
    #generate xmesh
    Q,P = generate_mesh(x_samples,qmax=4)
    xinit = torch.tensor(np.column_stack((Q.flatten(),P.flatten())),dtype=torch.float32,requires_grad=False)
    
    #intiialise pdf 
    pdf_out = torch.zeros_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32)

    with torch.no_grad():
        for _ in range(mc_samples):
            pdf_out += integrator.compute_pdf(xinit.detach().clone(),Tf)
    
    return xinit.detach().numpy(),pdf_out.detach().numpy()/mc_samples

class PlotUtils():
    """
    Class which prepares collates all requirements, such as the neural net, boundary conditions, reference and perturbative solutions, and plot functions
    for plotting 
    
    Functionality:
    1. sets filepaths to neural network checkpoints and loads them if necessary
    2. initialises plotter based on boundary conditions given either in parameter file or through cmd 
    2a. sets matplotlib parameters
    3. sets the location of the reference solution based on input folder
    4. creates results folders
    5. initialises the integrator to use for the the plots
    6. adds reference solutions if they are available
    """

    def __init__(self,folder=None,epoch=None):
        
        update_mpl()

        #get folder and load boundary data and params from file
        if folder is None:
            folder = fetch_results_folder_from_cmd()
        else:
            folder = folder 
        
        self.folder = folder

        boundary_cond,data_loaded = init_params_from_file(folder)

        params = checkpoint_plot_params() #get checkpoint plot parameters from cmd line
        
        self.boundary_cond = boundary_cond
        self.data_loaded = data_loaded

        if params["mcsamples"]>0:
            self.mc_samples = params["mcsamples"]
        else: #use test samples if no value is specified
            self.mc_samples = self.data_loaded["testsamples"]
        
        if epoch is None:
            self.epoch = params["epoch"]
        else:
            self.epoch = epoch
        
        self.plotter = PlotFuns(boundary_cond)

        self.boundary_cond.overwrite_ref_path(self.folder)
            
        self.referencesol = RefSolution(boundary_cond,folder_name=self.folder)
        
        #make new folder for the results labelled with epoch
        os.makedirs(self.folder+f"/plots_{self.epoch}", exist_ok = True) 

        self.resultsdir = self.folder+f"/plots_{self.epoch}"
        self.plotter.setresultsdir(self.resultsdir)

    def checkpoint_plots_du(self):
        """
        Runs the functions to make plots to show the results of the 
        drift learning
        """

        self.load_du_checkpoint()

        self.plotter.plot_drift(times = [0,self.boundary_cond.params["Tf"]/2,self.boundary_cond.params["Tf"]],net=self.net,refsol=self.referencesol)
        self.make_sde_plots()
        self.make_intermediate_histograms(plottimes= [0,1,2,3,4,5])

    def perturbative_plots_du(self,plottimes=[0,5]):
        """
        Runs the functions to make plots to show the results of the 
        evolution under perturbative drift 
        """
        
        self.load_perturbative_du_checkpoint()
        
        #self.make_intermediate_histograms(plottimes=plottimes)
        self.make_contour_plot2(plottimes=plottimes)
        self.plotter.plot_drift(times = plottimes,net=self.net,refsol=self.referencesol)#[0,self.boundary_cond.params["Tf"]/2,self.boundary_cond.params["Tf"]],net=self.net)

    def make_sde_plots(self):
            
        final_plot_time = self.boundary_cond.params["Tf"]
        
        t_vec, xinit, xref, cumulants_nn, cumulants_ref = self.integrator.compute_sde(final_plot_time,self.referencesol.duref_net)
        
        self.plotter.update_final_cumulants(cumulants_nn[-1,:])
        self.plotter.make_marginal_distribution_fig([xinit,xref],final_plot_time)
        
        fig = self.plotter.plot_cumulants(t_vec,[cumulants_nn,cumulants_ref],["NN estimated","Perturbative"],self.data_loaded["testsamples"])
        self.plotter.save_fig_(fig,"cumulants")
        self.plotter.save_cumulants_csv(t_vec,cumulants_nn)
        self.plotter.save_cumulants_csv(t_vec,cumulants_ref,cumulants_filename="ref_cumulants")

    def load_perturbative_du_checkpoint(self):

        self.net = self.referencesol.duref_net #NeuralNetwork(self.boundary_cond,self.data_loaded)
        #self.net.load_checkpoint(self.folder+f"/checkpoints/{self.net.net_name}_checkpoint_{self.epoch}.pt")
        self.integrator = langevin.SDE(self.boundary_cond,terminal_cost=None,net=self.net,sde_params=self.data_loaded)

    def load_du_checkpoint(self):
        self.net = NeuralNetwork(self.boundary_cond,self.data_loaded)
        self.net.load_checkpoint(self.folder+f"/checkpoints/{self.net.net_name}_checkpoint_{self.epoch}.pt")
        self.integrator = langevin.SDE(self.boundary_cond,terminal_cost=None,net=self.net,sde_params=self.data_loaded)

    def load_rho_checkpoint(self):
        
        self.rhonet = RhoNeuralNetwork(self.data_loaded)
        self.rhonet.load_checkpoint(self.folder+f"/checkpoints/{self.rhonet.net_name}_checkpoint_{self.epoch-1}.pt")
            
    def make_girsanov_net_plots(self,):
        self.load_rho_checkpoint()
        self.make_contour_plot()

    def make_intermediate_histograms(self,plottimes):

        #make figure object
        fig,axs = self.plotter.make_fig_subplots(ncols=int(np.ceil(len(plottimes)/2)),nrows=2)
        
        axs = axs.flatten()

        for i in range(len(plottimes)):
            
            xvals= self.integrator.update_ref_statistics(plottime=plottimes[i])
            
            h,l = self.plotter.marginal_distribution(axs[i],[xvals.detach().numpy()],plottimes[i],refsol=self.referencesol)
            
        #make legend 
        legend = fig.legend(h,l,ncols= 4, 
                                frameon=False,handlelength = 1)

        #move the legend
        legend.set_bbox_to_anchor(bbox=(0.95,0.07))
        
        fig.tight_layout()
        
        #fig.savefig(self.resultsdir+"/histograms_all.png",bbox_inches="tight")     
        self.plotter.save_fig_(fig,"histograms_all")
        plt.close()
    
    def checkpoint_plots_rho(self):
        
        self.load_rho_checkpoint()
        self.make_contour_plot()
        #self.plot_pdf_3d()
    
    def compute_rho_nn(self,x_samples=None,*_):
        """
        Generates a statespace grid and evaluates the neural network estimate of the 
        joint probability distribution at final time 

        args:
            x_samples : ndarray
                number of space discretisation steps
        returns:
            xout : ndarray
                default mesh of points of the statespace       
            jointout
                output of the neural network joint distribution at final time
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

    def make_contour_plot2(self,plottimes = [0,1,2,3,4,5]):
        
        ncols = int(np.ceil(len(plottimes)/2))
        fig,axs = self.plotter.make_fig_subplots(ncols=ncols,nrows=2)
        axs = axs.flatten()

        for i in range(len(plottimes)):

            xout,jointout = compute_rho_girsanov(self.integrator,x_samples=50,mc_samples=self.mc_samples,Tf=plottimes[i])

            self.plotter.plot_joint_panel(axs[i],plottimes[i],jointout,xout,refsol=self.referencesol)
        
        fig.subplots_adjust(hspace=1.1,wspace=1,bottom=0.08)
        
        #add legend
        h,l = self.plotter.get_all_subplot_handles_labels(fig)
        self.plotter.make_legend(fig,h,l)
        self.plotter.make_colourbar(fig,vmax=0.5)
        
        self.plotter.save_fig_(fig,"contour_distribution")
        
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
        plt.savefig(self.resultsdir+"/test_3d.png",bbox_inches="tight")        

    def make_contour_plot(self):
        xout,jointout = self.compute_rho_nn(x_samples=20)
        
        fig,ax = plt.subplots()
        ax, ax_pmarginal,ax_qmarginal = self.plotter.scatter_panel_layout(ax)

        self.plotter.joint_distributions_scatter_panel(ax,
                                                    ax_pmarginal,
                                                    ax_qmarginal,
                                                    jointout,
                                                    xout,
                                                    time=self.boundary_cond.params["Tf"],vmax=0.5,refsol=self.referencesol)
        
        self.plotter.make_legend(fig,h,l)
        self.plotter.make_colourbar(fig,vmax=0.5)
        
        self.plotter.save_fig_(fig,"contour_distribution")
        
        plt.close()
        

def compute_ref_dynamic_kl(boundary):
    """Estimate of the dynamic kl divergence for the Gaussian reference data.
    Uses the reference position and momentum variance and estimates the integral
    E_(P) (du)**2 in the interval [0,T]
    where the (P) is the path measure estimated with a Gaussian at each step 
    """ 
    
    times = boundary.get_reference_timegrid()
    klout = np.empty(len(times))
    #mean_t = boundary.get_known_solution_cumulants("Position Mean")
    var_t = boundary.get_known_solution_cumulants("Position Variance")
    for t in enumerate(times):
        xref,duref = boundary.get_reference_data(t[1])
        ptx = plotter.normal_distribution(xref,0,var_t[t[0]])
        klout[t[0]] = np.trapz((duref**2)*ptx,xref)

    return np.trapz(klout,times)

if __name__=="__main__":

    plot_utils = PlotUtils()

    plot_utils.checkpoint_plots_du()