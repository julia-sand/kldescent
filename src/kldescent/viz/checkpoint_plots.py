import os
import yaml 

from mpl_toolkits import mplot3d

import torch
import numpy as np

from kldescent.model.nn import RhoNeuralNetwork,NeuralNetwork,load_checkpoint
from kldescent.sde import langevin
from kldescent.sde.girsanovpdf import compute_pdf
from kldescent.utils.parser import checkpoint_plot_params,fetch_results_folder_from_cmd,metric_params
from kldescent.utils.getboundary import init_params_from_file
from kldescent.benchmark.benchmarksol import PerturbativeBenchmarkSolution,GaussBenchmarkSolution

from kldescent.viz.plots import *
from kldescent.viz.cumulants import plot_cumulants
from kldescent.viz.drift import plot_drift
from kldescent.viz.distribution_plots import marginal_distribution,make_marginal_distribution_fig


def zchop(x,tol=1e-6):
    return np.where(np.abs(x)<tol,0,x)

@torch.no_grad
def generate_mesh(x_samples,qmax=6):
    """
    generates a mesh in two dimensional phase space to evaluate and plot the joint distribution 
    
    args:
        x_samples:
            number of samples in the space discretisation
    """

    p_init = torch.linspace(-qmax,qmax,x_samples, requires_grad=False,dtype=torch.float32)
    #q_init = torch.linspace(-qmax,qmax,x_samples, requires_grad=False,dtype=torch.float32)
        
    return torch.meshgrid(p_init,p_init,indexing='ij')

@torch.no_grad
def compute_rho_girsanov(x_samples=40,boundary=None,net=None,mc_samples=64,Tf=None):
    """
    Computes an estimate of the final joint distribution using girsanov theorem
    under the input drift 
    
    args:
        dunet : class 
            neural network approximating the drift
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
    Q,P = generate_mesh(x_samples,qmax=5)
    xinit = torch.stack((X, Y), dim=-1).reshape(-1, 2)
    #torch.tensor(np.column_stack((Q.flatten(),P.flatten())),dtype=torch.float32,requires_grad=False)
    
    with torch.no_grad():
        self.net.eval()

        #intiialise pdf 
        pdf_out = torch.zeros_like(xinit[...,[0]],requires_grad=False,dtype=torch.float32)

        pdf_out = compute_pdf(xinit.clone(),net,
                                    self.boundary.sigma0,
                                    self.boundary.params["epsilon"],
                                    self.boundary.p_initial,
                                    self.boundary.params["Tf"]).cpu().numpy()
    return xinit.cpu().numpy(),pdf_out.cpu().numpy()

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

    def __init__(self,epoch=None,benchmark=None,folder=None,wandb_run=None):
        """
        args:
            epoch : int
                Epoch which will be plotted. Default to command line argument.
            benchmark : class
                Default None. If passed, plots will include the benchmarks 
            folder : str
                Default to command line argument. If passed, overwrites cmd line and uses different folder to find the checkpoints and save plots
            wandbrun :
                Default None. If passed, plots are saved in the wandb run 
        """
        
        update_mpl()

        #get folder and load boundary data and params from file
        if folder is None:
            self.folder = fetch_results_folder_from_cmd()
        else:
            self.folder = folder 
        
        boundary_cond,data_loaded = init_params_from_file(self.folder)

        params = checkpoint_plot_params() #get checkpoint plot parameters from cmd line
        
        self.boundary_cond = boundary_cond
        self.data_loaded = data_loaded

        try:
            if params["mcsamples"]>0:
                self.mc_samples = params["mcsamples"]
            else: #use test samples if no value is specified
                self.mc_samples = self.data_loaded["testsamples"]
        except TypeError:
            self.mc_samples = 20 #overwrite if not passed or no loaded data   

        if epoch is None:
            self.epoch = params["epoch"]
        else:
            self.epoch = epoch
       
        #self.plotter = PlotFuns(boundary_cond)

        self.boundary_cond.overwrite_ref_path(self.folder)
            
        self.benchmark = benchmark
                
        #make new folder for the results labelled with epoch
        os.makedirs(self.folder+f"/plots_{self.epoch}", exist_ok = True) 

        self.resultsdir = self.folder+f"/plots_{self.epoch}"

        self.wandb_run = wandb_run

    def load_perturbative_du_checkpoint(self):

        if self.benchmark is None:
            #initialise benchmarks
            self.benchmark = PerturbativeBenchmarkSolution(boundary_cond,folder_name=self.folder)
            self.net = self.benchmark.du_net_benchmark 

            self.integrator = langevin.VerletSDE(self.boundary_cond,terminal_cost=None,net=self.net,sde_params=self.data_loaded)



    def load_du_checkpoint(self):
        """Load neural network and initialise integration method"""

        self.net = NeuralNetwork(self.boundary_cond,nn_params=self.data_loaded)
        load_checkpoint(self.net,self.folder+f"/checkpoints/{self.net.net_name}_checkpoint_{self.epoch}.pt")
        
        self.integrator = langevin.EulerMaruyamaSDE(self.boundary_cond,terminal_cost=None,net=self.net,sde_params=self.data_loaded)

    def load_rho_checkpoint(self):
        
        self.rhonet = RhoNeuralNetwork(self.data_loaded)
        load_checkpoint(self.rhonet,self.folder+f"/checkpoints/{self.rhonet.net_name}_checkpoint_{self.epoch}.pt")
            
    def checkpoint_plots_du(self,usebenchmark=False,mcsamples=None):
        r"""
        Make plots to show the results of the 
        drift learning
        params: 
            usebenchmark : bool
                If true, load reference solution, else pass
            mcsamples : int
                Overwrite default number of test/mc samples
        """
        
        self.load_du_checkpoint()
        if usebenchmark:
            self.load_benchmark_solution()
        else:
            self.benchmark=None

        self.make_drift_plot(numplots=4)
        self.make_sde_plots(mcsamples,usebenchmark)
        #self.make_intermediate_histograms()
        
        plt.close("all")

    def load_benchmark_solution(self):
        """initialise the reference solution and add to the class""" 

        if self.benchmark is None:
            if "Gauss" in f"{self.boundary_cond.__class__.__name__}":  
                self.benchmark = GaussBenchmarkSolution(boundary=self.boundary_cond,
                                                            folder_name=self.folder)
            
            else:
                self.benchmark = PerturbativeBenchmarkSolution(boundary=self.boundary_cond,
                                                                folder_name=self.folder)
            
    def perturbative_plots_du(self,plottimes):
        """make plots to show the results of the evolution under perturbative drift"""
        
        self.load_perturbative_du_checkpoint()
        
        #self.make_intermediate_histograms()
        self.make_drift_plot(numplots=4)

    @torch.no_grad
    def make_sde_plots(self,mcsamples,usebenchmark):
        """Make plots and save results of sde integration 
        1. compute final point clouds
        2. make final time marginal distribution plot
        3. compute cumulants along trajectories 
        4. make cumulant plots
        5. save csv of reference and computed csvs. 
        """

        if mcsamples is None: #set number of samples if not passed
            mcsamples = self.data_loaded["testsamples"]

        final_plot_time = self.boundary_cond.params["Tf"]

        #try:
        #    t_vec,cumulants_nn,cumulants_ref = self.load_cumulant_csvs()
        #except FileNotFoundError:
        t_vec, xinit, xref, cumulants_nn, cumulants_ref = self.integrator.compute_sde_cumulants(benchmark=self.benchmark,mcsamples=mcsamples)
            
        #marginal distribution fig
        fig = make_marginal_distribution_fig([xinit,xref],final_plot_time,benchmark=self.benchmark)
        save_fig_(fig,f"marginal_time_{final_plot_time}",loc=self.resultsdir)
        
        self.make_cumulant_plot(t_vec,cumulants_nn,cumulants_ref,mcsamples,
                                        add_free_diffusion=False,)
        if usebenchmark:
            self.make_cumulant_error_plot(t_vec,cumulants_nn,cumulants_ref)
        
        plt.close("all")
    
    @torch.no_grad
    def evaluate_net_output(self,net,time):
        """Evaluate neural network at time on 100 space coordinates 
        args:
            net : class
                neural network to evaluate
            time : float

        """

        net.eval()
        with torch.no_grad():
            #computes neural network output (forward pass) to use in plotting
            return net(torch.cat((time*torch.ones((1000,1),requires_grad=False,dtype=torch.float32).to(net.device)
                                                        ,torch.linspace(-self.boundary_cond.qmax,self.boundary_cond.qmax,1000,requires_grad=False).unsqueeze(1).to(net.device)),dim=-1)).cpu().numpy()
        
    def generate_times(self,numplots):
        
        if self.benchmark is not None:

            if self.benchmark.du_net_benchmark is None:
                #get the timegrid where data is available
                t_vec = self.benchmark.get_timegrid()
                #choose numplot random times to show
                times = t_vec[np.random.randint(low=0,high=len(t_vec),size=numplots)]
                return np.sort(times)
            
            else:
                times = np.sort(np.random.uniform(0, self.boundary_cond.params["Tf"],numplots))
                return times
        else:
            times = np.sort(np.random.uniform(0, self.boundary_cond.params["Tf"],numplots))
            return times
    
    def make_drift_plot(self,numplots):
    
        times = self.generate_times(numplots)
        #[0,1.100000,2.300000,4.150000,5]
        
        net_vals = self.get_drift_data(times)
        ref_vals = self.get_benchmark_drift_data(times)

        fig = plot_drift(times,
                            net_vals,
                            ref_vals)
        
        if self.wandb_run is not None:
            self.wandb_run.log({"drift":fig})

        save_fig_(fig,"drift",self.resultsdir)
        plt.close()
        return
    

    def evaluate_net_output_inverted(self,net,xcoord):
        return net(torch.cat((torch.linspace(0,self.boundary_cond.params["Tf"],1000,requires_grad=False).unsqueeze(1).to(net.device),
                        xcoord*torch.ones((1000,1),requires_grad=False,dtype=torch.float32).to(net.device)),dim=-1)).cpu().numpy()
       


    def get_benchmark_drift_data_inverted(self,xcoords):
        if self.benchmark is None:
            return None
        else:
            benchmark_vals = []
            for xcoord in xcoords:
                benchmark_vals.append(self.evaluate_benchmark_drift_inverted(xcoord))

            return benchmark_vals

    def evaluate_benchmark_drift_inverted(self,xcoord):
    
        if self.benchmark.du_net_benchmark is not None:
            return self.evaluate_net_output_inverted(self.benchmark.du_net_benchmark,xcoord) 
            #self.benchmark.du_net_benchmark(torch.cat((torch.linspace(0,self.boundary_cond.params["Tf"],1000,requires_grad=False).unsqueeze(1).to(self.net.device),
                    #         xcoord*torch.ones((1000,1),requires_grad=False,dtype=torch.float32).to(self.net.device)),dim=-1)).cpu().numpy()
        else:
            return (-1)*xcoord*np.ones(1000)*self.boundary_cond.params["epsilon"]*self.benchmark.get_underdamped_drift_array()[:,1]/np.sqrt(2)


    def make_inverted_drift_plot(self,numplots):
        """show the drift as a function of time at different values of x
        """
    
        xcoords = np.sort(np.random.uniform(-5,5,numplots))
                
        self.net.eval()

        with torch.no_grad():
            benchmark_vals = self.get_benchmark_drift_data_inverted(xcoords)
            net_vals = self.get_drift_data_inverted(xcoords)
        
        fig = plot_drift(xcoords,
                            net_vals,
                            benchmark_vals,
                            xaxis=np.linspace(0,self.boundary_cond.params["Tf"],1000),
                            invert_labels=True)
        
        if self.wandb_run is not None:
            self.wandb_run.log({"drift":fig})

        save_fig_(fig,"drift_inverted",self.resultsdir)
        plt.close()
        return

    @torch.no_grad
    def get_benchmark_drift_data(self,times):
    
        if self.benchmark is None:
            ref_vals = None
        
        else:
            #initialise the arrays
            ref_vals = []

            for t in range(len(times)):
            
                if self.benchmark.du_net_benchmark is not None:
                    ref_vals.append(self.evaluate_net_output(self.benchmark.du_net_benchmark,times[t]).flatten())
                else: 
                    ref_vals.append(self.benchmark.get_drift_gaussian(times[t])*np.linspace(-self.boundary_cond.qmax,self.boundary_cond.qmax,1000))

        return ref_vals

    @torch.no_grad
    def get_drift_data(self,times):
        
        #initialise the arrays
        net_vals = []

        for t in range(len(times)):
            
            net_vals.append(self.evaluate_net_output(self.net,times[t]).flatten())
        
        return net_vals
    
    def get_drift_data_inverted(self,xcoords):

        net_vals = []
        for xcoord in xcoords:    
            #computes neural network output (forward pass) to use in plotting
            net_vals.append(self.evaluate_net_output_inverted(self.net,xcoord).flatten())

        return net_vals 

    def compute_drift_errs(self,time):

        net_val,ref_val = self.get_drift_data([time])
        l1 = np.mean(self.l1_error(net_val[0],ref_val[0]))
        l2 = np.sqrt(np.mean(self.l2_error(net_val[0],ref_val[0])))
        
        return l1,l2

    def make_rhonet_marginals_plot(self):
        """Make plot and add legend"""
        fig = self.plot_rhonet_marginals()

        h,l = get_all_subplot_handles_labels(fig)

        make_legend(fig,h,l)

        if self.wandb_run is not None:
            self.wandb_run.log({"rhonetmargs":fig})

        fig.tight_layout()
        save_fig_(fig,"rho_marginals",loc=self.resultsdir)
        plt.close("all")

    def plot_rhonet_marginals(self):
        """compute rho data and plot"""

        self.load_rho_checkpoint()
        #generate empty figure
        fig = make_marginal_distribution_fig(xvals=[],plottime=self.boundary_cond.params["Tf"])

        ax0 = fig.get_axes()
        ax_list = get_inset_axes(ax0[0])
        
        for compute_rho_fun,label in zip([self.compute_rho_nn,compute_rho_girsanov],["girsanovnet","mc"]):
            xs,joint_out = compute_rho_fun(x_samples=20,boundary=self.boundary_cond,net=self.net)

            pinit,qmargnorm,pmargnorm,_ = compute_marginals_from_joint(xs,joint_out)
            
            ax_list[0].plot(pinit,qmargnorm,label=label)
            ax_list[1].plot(pinit,pmargnorm,label=label) 
       
        return fig

    def make_girsanov_net_plots(self,):
        
        self.load_rho_checkpoint()
        self.load_du_checkpoint()
        #self.make_contour_plot()
       
        self.make_rhonet_marginals_plot()
        plt.close("all")

    def make_intermediate_histograms(self):#,plottimes):
        
        plottimes = self.generate_times(6)

        #make figure object
        fig,axs = make_fig_subplots(ncols=int(np.ceil(len(plottimes)/2)),nrows=2)
        
        axs = axs.flatten()

        for i in range(len(plottimes)):
            
            self.integrator.net = self.net
            x_du = self.integrator.update_ref_statistics(time=plottimes[i]).cpu().numpy()
            print(x_du.shape)
            if self.benchmark is not None:

                if self.benchmark.du_net_benchmark is not None:  
                    #UPDATE NET FOR INTEGRATOR
                    self.integrator.net = self.benchmark.du_net_benchmark
                    x_ref = x_du#self.integrator.update_ref_statistics(time=plottimes[i]).cpu().numpy()
                else:
                    x_ref = np.zeros_like(x_du)
                    #x_ref[:,[0]] = self.benchmark.get_benchmark_quantity(plottimes[i],"underdamped marginal", data="perturbative underdamped")[1,np.newaxis]
                    #print(x_ref.shape)
            
            marginal_distribution(axs[i],[x_du,x_ref],plottimes[i],benchmark=self.benchmark)
            
        h,l = get_all_subplot_handles_labels(fig)
        
        #make the legend 
        legend = fig.legend(h,l,ncols= 4, 
                                frameon=False,handlelength = 1)

        #move the legend
        legend.set_bbox_to_anchor(bbox=(0.95,0.07))
        
        fig.tight_layout()
        
        #if self.wandb_run is not None:
        #    self.wandb_run.log({"inthists":fig})

        save_fig_(fig,"histograms_all",loc=self.resultsdir)
        plt.close()
    
    def checkpoint_plots_rho(self):
        
        self.load_rho_checkpoint()
        #self.make_contour_plot()
        print("NO PLOTS WILL BE OUTPUT. UPDATE THIS SECTION")
        plt.close("all")

    
    def compute_rho_nn(self,x_samples=None,**kwargs):
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
            xinit = torch.stack((X, Y), dim=-1).reshape(-1, 2)
            
            pdf_out = self.rhonet(xinit.detach().clone())

            return xinit.detach().numpy(),pdf_out.detach().numpy()

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
        plt.close("all")


    def l1_error(self,test_vec,ref_vec):
        """dinf"""
        return zchop(np.abs(test_vec-ref_vec))/np.maximum(np.abs(ref_vec),np.abs(test_vec))
        #np.zeros_like(cumulants_nn)#np.abs(zchop(cumulants_nn)-zchop(cumulants_ref))/zchop(cumulants_ref)

    def l2_error(self,test_vec,ref_vec):

        return zchop(np.abs(test_vec-ref_vec)**2)/np.maximum(np.abs(ref_vec),np.abs(test_vec))

    def load_cumulant_csvs(self,):
        """return tvec, nn cumulants and benchmark cumulants from csvs"""

        cumulants_nn = np.loadtxt(open(self.resultsdir+"/cumulants.csv"),
                        delimiter=",",skiprows=1) 
        cumulants_ref = np.loadtxt(open(self.resultsdir+"/ref_cumulants.csv"),
                        delimiter=",",skiprows=1)

        return cumulants_nn[:,0],cumulants_nn[:,1:],cumulants_ref[:,1:]
    
    def make_cumulant_error_plot_from_csv(self,):
        #mcsamples = self.data_loaded["testsamples"]
        t_vec,cumulants_nn,cumulants_ref = self.load_cumulant_csvs()
        self.make_cumulant_plot(t_vec,cumulants_nn,cumulants_ref,mcsamples=0)
        self.make_cumulant_error_plot(t_vec,cumulants_nn,cumulants_ref)

    def make_cumulant_error_plot(self,t_vec,cumulants_nn,cumulants_ref):    
        l1_err = self.l1_error(cumulants_nn,cumulants_ref)
        l2_err = self.l2_error(cumulants_nn,cumulants_ref)
        fig = plot_cumulants(t_vec,[l1_err,l2_err],
                                                        ["Relative L1","Relative L2"],
                                                        0,
                                                        benchmark=None,
                                                        add_free_diffusion=False)
        save_fig_(fig,"cumulants_errors",loc=self.resultsdir)
        
        #if self.wandb_run is not None:
        #    self.wandb_run.log({"cumerrs":fig})

    def make_drift_error_plot(self):    

        self.load_du_checkpoint()
        self.load_benchmark_solution()

        with torch.no_grad():
            t_vec = self.benchmark.get_timegrid()[::10]#compute du 

            l1_errs = np.zeros_like(t_vec)
            l2_errs = np.zeros_like(t_vec)
            
            for i in range(len(t_vec)):
                l1_errs[i],l2_errs[i] = self.compute_drift_errs(t_vec[i])
            
        fig = plt.figure()
        plt.plot(t_vec,l1_errs,label="L1")
        plt.plot(t_vec,l2_errs,label="L2")
        plt.xlabel("t")
        plt.ylabel("Error")
        plt.legend()
        print("L1 = ",np.trapezoid(l1_errs,t_vec))
        print("L2 = ",np.trapezoid(l2_errs,t_vec))
        save_fig_(fig,"drift_errors",loc=self.resultsdir)

    def make_cumulant_plot(self,t_vec,cumulants_nn,cumulants_ref,
                            mcsamples,
                            add_free_diffusion=False):

        #cumulant figs and csv 
        fig = plot_cumulants(t_vec,[cumulants_nn,cumulants_ref],
                                                ["NN estimated","Benchmark"],
                                                mcsamples,
                                                self.benchmark,
                                                add_free_diffusion,
                                                self.boundary_cond.params["epsilon"])
        if self.wandb_run is not None:
            self.wandb_run.log({"cumulants":fig})

        save_fig_(fig,"cumulants",loc=self.resultsdir)
        save_cumulants_csv(t_vec,cumulants_nn,loc=self.resultsdir)
        save_cumulants_csv(t_vec,cumulants_ref,loc=self.resultsdir,cumulants_filename="ref_cumulants")
        plt.close("all")

def compute_ref_dynamic_kl(benchmark):
    """Estimate of the dynamic kl divergence for the Gaussian benchmark data.
    Uses the benchmark position and momentum variance and estimates the integral
    E_(P) (du)**2 in the interval [0,T]
    where the (P) is the path measure estimated with a Gaussian at each step 
    
    ##TODO: update this for non-gaussian

    args:
        benchmark : class
            specify the benchmark solution for comparison
    returns: 
        kldyn : float
            estimate of the kl divergence computed for the reference solution  
    """ 
    
    times = benchmark.get_timegrid() 
    klout = np.empty(len(times))

    var_t = boundary.get_known_solution_cumulants("Position Variance")

    for t in enumerate(times):
        xref,duref = boundary.get_reference_data(t[1])
        ptx = normal_distribution(xref,0,var_t[t[0]])
        klout[t[0]] = np.trapezoid((duref**2)*ptx,xref)

    return np.trapezoid(klout,times)

if __name__=="__main__":

    plot_utils = PlotUtils()
    usebenchmark,_ = metric_params()

    plot_utils.load_du_checkpoint()
    if usebenchmark:
        plot_utils.load_benchmark_solution()

    plot_utils.make_drift_plot(numplots=5)
    plot_utils.make_inverted_drift_plot(numplots=5)
    plot_utils.make_sde_plots(2048, usebenchmark)
    #plot_utils.checkpoint_plots_du(usebenchmark=usebenchmark)
    #plot_utils.make_cumulant_error_plot_from_csv()
    #plot_utils.make_intermediate_histograms()
    #plot_utils.make_cumulant_error_plot_from_csv()

    if plot_utils.data_loaded["terminalcost"]=="GirsanovNet":
        plot_utils.make_girsanov_net_plots()