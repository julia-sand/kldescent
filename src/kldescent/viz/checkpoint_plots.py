import os

import torch
import numpy as np

from kldescent.model.nn import RhoNeuralNetwork,NeuralNetwork,load_checkpoint
from kldescent.sde import langevin
from kldescent.utils.setseed import set_seed
from kldescent.utils.parser import checkpoint_plot_params,fetch_results_folder_from_cmd,metric_params
from kldescent.utils.getboundary import init_params_from_file
from kldescent.benchmark.benchmarksol import PerturbativeBenchmarkSolution,GaussBenchmarkSolution
from kldescent.viz.contour import *

from kldescent.viz.plots import *
from kldescent.viz.cumulants import plot_cumulants,load_cumulant_csvs
from kldescent.viz.drift import plot_drift
from kldescent.viz.distribution_plots import marginal_distribution,make_marginal_distribution_fig

def zchop(x,tol=1e-8):
    return np.where(np.abs(x)<tol,0,x)

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

        #set useful parameters and defaults if they are not in param file
        try:
            self.bridge_type = data_loaded["bridgetype"]
            self.loss_type = data_loaded.get("loss", "kl")

            if params["mcsamples"]>0:
                self.mc_samples = params["mcsamples"]
            else:
                self.mc_samples = self.data_loaded["testsamples"]

        except TypeError: #missing entries 
            self.bridge_type = "half"
            self.loss_type = "kl"
            self.mc_samples = 20

        if epoch is None:
            self.epoch = params["epoch"] if params["epoch"] is not None else self.select_best_epoch()
        else:
            self.epoch = epoch
       
        self.boundary_cond.overwrite_ref_path(self.folder)
            
        self.benchmark = benchmark

        try:
            set_seed(self.data_loaded["seed"])
        except:
            set_seed()
    
        #make new folder for the results labelled with epoch
        os.makedirs(self.folder+f"/plots_{self.epoch}", exist_ok = True) 

        self.resultsdir = self.folder+f"/plots_{self.epoch}"

        self.wandb_run = wandb_run

    def load_perturbative_du_checkpoint(self,):

        #if self.benchmark is None:
        self.load_benchmark_solution(self.folder)
        #initialise benchmarks
        #self.benchmark = PerturbativeBenchmarkSolution(bridge_type="half", boundary=self.boundary_cond, folder_name=self.folder)
        #self.benchmark.initialise_net(self.benchmark.bridge_type)

        #set the net as the benchmark perturbative net 
        self.net = self.benchmark.du_net_benchmark 

        #create integrator
        integrator_cls = getattr(langevin, self.data_loaded.get("integrator", "BAOAB"))
        self.integrator = integrator_cls(self.boundary_cond, net=self.net, sde_params=self.data_loaded)

    def load_du_checkpoint(self):
        """Load neural network and initialise integration method"""

        self.net = NeuralNetwork(self.boundary_cond,nn_params=self.data_loaded)
        load_checkpoint(self.net,self.folder+f"/checkpoints/{self.net.net_name}_checkpoint_{self.epoch}.pt")

        integrator_cls = getattr(langevin, self.data_loaded.get("integrator", "BAOAB"))
        self.integrator = integrator_cls(self.boundary_cond, net=self.net, sde_params=self.data_loaded)

    def load_rho_checkpoint(self):
        
        self.rhonet = RhoNeuralNetwork(self.data_loaded)
        load_checkpoint(self.rhonet,self.folder+f"/checkpoints/{self.rhonet.net_name}_checkpoint_{self.epoch}.pt")
            
    def checkpoint_plots_du(self,usebenchmark=None,mcsamples=None):
        r"""
        Make plots to show the results of the 
        drift learning
        params: 
            usebenchmark : bool
                path to benchmark solution
            mcsamples : int
                Overwrite default number of test/mc samples
        """

        self.load_du_checkpoint()
        if usebenchmark is not None:
            self.load_benchmark_solution(usebenchmark)
        else:
            self.benchmark=None

        self.make_drift_plot(numplots=4)
        self.make_sde_plots()
        #self.make_intermediate_histograms()

        plt.close("all")

    def load_benchmark_solution(self,usebenchmark):
        """initialise the reference solution and add to the class""" 

        if self.benchmark is None:

            if "Gauss" in f"{self.boundary_cond.__class__.__name__}":
                self.benchmark = GaussBenchmarkSolution(boundary=self.boundary_cond,
                                                        folder_name=usebenchmark,
                                                        loss_type=self.loss_type)

            else:
                self.benchmark = PerturbativeBenchmarkSolution(bridge_type=self.bridge_type,
                                                               boundary=self.boundary_cond,
                                                               folder_name=usebenchmark,
                                                               loss_type=self.loss_type)
                self.benchmark.initialise_net(self.benchmark.bridge_type, nn_params=self.data_loaded)
                #self.net = self.benchmark.du_net_benchmark 

    def perturbative_plots_du(self,plottimes):
        """make plots to show the results of the evolution under perturbative drift"""

        self.load_perturbative_du_checkpoint()

        self.make_intermediate_histograms()
        self.make_drift_plot(numplots=4)

    @torch.no_grad
    def make_sde_plots(self,):
        """Make plots and save results of sde integration
        1. compute final point clouds
        2. make final time marginal distribution plot
        3. compute cumulants along trajectories
        4. make cumulant plots
        5. save csv of reference and computed csvs.
        6. save csv of samples from final distribution
        """

        #if mcsamples is None: #set number of samples if not passed
        mcsamples = self.mc_samples #self.data_loaded["testsamples"]

        final_plot_time = self.boundary_cond.params["Tf"]

        t_vec, xinit, xref, cumulants_nn, cumulants_ref = self.integrator.compute_sde_cumulants(benchmark=self.benchmark,mcsamples=mcsamples)

        #save final samples (xinit) to csv
        np.savetxt(f"{self.resultsdir}/samples_nn_t{final_plot_time}.csv",
                xinit, delimiter=",", header="q,p", comments="")

        #marginal distribution fig
        fig = make_marginal_distribution_fig([xinit,xref],final_plot_time,benchmark=self.benchmark)
        save_fig_(fig,f"marginal_time_{final_plot_time}",loc=self.resultsdir)

        self.make_cumulant_plot(t_vec,t_vec,cumulants_nn,cumulants_ref,mcsamples,
                                        add_free_diffusion=False,)
        if self.benchmark is not None:
            self.make_cumulant_error_plot(t_vec,t_vec,cumulants_nn,cumulants_ref)

        self.plot_final_position_marginals(xinit)
        plt.close("all")

    @torch.no_grad
    def make_cumulant_plot_from_csv(self):
        t_vec1,t_vec2,cumulants_nn,cumulants_ref = load_cumulant_csvs(self.folder)

        self.make_cumulant_plot(t_vec1,t_vec2,cumulants_nn,cumulants_ref,mcsamples,
                                        add_free_diffusion=True,epsilon=self.boundary_cond.params["epsilon"])
        
        
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

            #if self.benchmark.du_net_benchmark is None:
            
            #get the timegrid where data is available
            t_vec = self.benchmark.get_timegrid()
            
            #choose numplot random times to show
            #times = t_vec[np.random.randint(low=0,high=len(t_vec),size=numplots)]
            #return np.sort(times)
            
            tgrid =self.benchmark.get_timegrid() 
            indices = np.linspace(0, len(tgrid) - 1, numplots, dtype=int)
            
            return tgrid[indices]

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
            
                temp = self.benchmark.du_benchmark(times[t],np.linspace(-self.boundary_cond.qmax,self.boundary_cond.qmax,1000))
                ref_vals.append(temp)

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
            xs,joint_out = compute_rho_fun(x_samples=20, integrator=self.integrator)

            q_vals,p_vals,qmargnorm,pmargnorm,_ = compute_marginals_from_joint(xs,joint_out)

            ax_list[0].plot(q_vals,qmargnorm,label=label)
            ax_list[1].plot(p_vals,pmargnorm,label=label)
       
        return fig

    def make_girsanov_net_plots(self,):
        
        self.load_rho_checkpoint()
        self.load_du_checkpoint()
        #self.make_contour_plot()
       
        self.make_rhonet_marginals_plot()
        plt.close("all")

    def make_intermediate_histograms(self,testing_samples=None):#,plottimes):
        
        if testing_samples is None:
            testing_samples = self.mc_samples

        plottimes = [0,1,2,3,4,5]#self.generate_times(6)

        #make figure object
        fig,axs = make_fig_subplots(ncols=int(np.ceil(len(plottimes)/2)),nrows=2)
        
        #adjust figsize
        fig.set_figheight(15)
        fig.set_figwidth(25)
        
        axs = axs.flatten()

        for i in range(len(plottimes)):
            
            self.integrator.net = self.net
            x_du = self.integrator.update_ref_statistics(time=plottimes[i],testing_samples=testing_samples).cpu().numpy()
            
            marginal_distribution(axs[i],[x_du],plottimes[i],benchmark=self.benchmark)
            
        h,l = get_all_subplot_handles_labels(fig)
        
        #make the legend 
        legend = fig.legend(h,l,ncols= 4, 
                                frameon=False,handlelength = 1)

        #move the legend
        legend.set_bbox_to_anchor(bbox=(0.95,0.07))
        
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        
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
        
            if x_samples is None:
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

    def make_cumulant_error_plot_from_csv(self,):
        #mcsamples = self.data_loaded["testsamples"]
        t_vec1,t_vec2,cumulants_nn,cumulants_ref = load_cumulant_csvs(folder=self.resultsdir)
        self.make_cumulant_plot(t_vec1,t_vec2,cumulants_nn,cumulants_ref,mcsamples=0)
        self.make_cumulant_error_plot(t_vec1,t_vec2,cumulants_nn,cumulants_ref)

    def make_cumulant_error_plot(self,t_vec1,t_vec2,cumulants_nn,cumulants_ref):    
        l1_err = self.l1_error(cumulants_nn,cumulants_ref)
        l2_err = self.l2_error(cumulants_nn,cumulants_ref)
        fig = plot_cumulants(t_vec1,t_vec2,[l1_err,l2_err],
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

    def make_cumulant_plot(self,t_vec1,t_vec2,cumulants_nn,cumulants_ref,
                            mcsamples,
                            add_free_diffusion=False):

        #cumulant figs and csv 
        fig = plot_cumulants(t_vec1,t_vec2,[cumulants_nn,cumulants_ref],
                                                ["NN estimated","Benchmark"],
                                                mcsamples,
                                                self.benchmark,
                                                add_free_diffusion,
                                                self.boundary_cond.params["epsilon"])
        if self.wandb_run is not None:
            self.wandb_run.log({"cumulants":fig})

        save_fig_(fig,"cumulants",loc=self.resultsdir)
        save_cumulants_csv(t_vec1,cumulants_nn,loc=self.resultsdir)
        save_cumulants_csv(t_vec2,cumulants_ref,loc=self.resultsdir,cumulants_filename="ref_cumulants")
        plt.close("all")

    def make_contour_plot2(self,numplots = 6):

        plottimes = self.generate_times(numplots)

        make_contour_plot_(integrator=self.integrator,
                            boundary=self.boundary_cond,
                            resultsdir=self.resultsdir,
                            plottimes=plottimes,
                            benchmark=self.benchmark,
                            mc_samples=self.mc_samples)


    def select_best_epoch(self,):
        print("No plotting epoch given. Automatically selecting epoch with lowest Action Cost.")
        arr = np.loadtxt(self.folder+"/loss.csv",delimiter=",",skiprows=1)  
        return int(arr[np.argmin(arr[:,2]),0])


    def plot_final_position_marginals(self,xinit):
        """
        Plot the final achieved position marginals vs the target position marginals for d=2.
        """
        if self.boundary_cond.d != 2:
            print(f"Skipping final position marginals plot: dimension is {self.boundary_cond.d}, expected 2.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        q_range = np.linspace(-5, 5, 500)
        
        for i in range(2):
            ax = axes[i]
            
            # Plot learned marginal
            # Using KDE for smooth visualization
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(xinit[:, i].reshape(-1, 1))
            log_dens = kde.score_samples(q_range.reshape(-1, 1))
            ax.plot(q_range, np.exp(log_dens), label='Learned', color='blue', linewidth=2)
            
            # Plot target marginal
            target_pdf = self.boundary_cond.q_marginal_final_np(q_range)
            ax.plot(q_range, target_pdf, label='Target', color='red', linestyle='--', linewidth=2)
            
            ax.set_title(f'Position Marginal $q_{i+1}$')
            ax.set_xlabel('$q$')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_fig_(fig, "final_position_marginals", loc=self.resultsdir)
        plt.close(fig)


def compute_ref_dynamic_kl_gauss(benchmark):
    """Estimate of the dynamic kl divergence for the Gaussian benchmark data.
    Uses the benchmark position and momentum variance and estimates the integral
    E_(P) (du)**2 in the interval [0,T]
    where the (P) is the path measure estimated with a Gaussian at each step 
    
    args:
        benchmark : class
            specify the benchmark solution for comparison
    returns: 
        kldyn : float
            estimate of the kl divergence computed for the reference solution  
    """ 
    
    times = benchmark.get_timegrid() 
    klout = np.empty(len(times))

    var_t = benchmark.boundary.get_known_solution_cumulants("Position Variance")

    for t in enumerate(times):
        xref,duref = benchmark.boundary.get_reference_data(t[1])
        ptx = normal_distribution(xref,0,var_t[t[0]])
        klout[t[0]] = np.trapezoid((duref**2)*ptx,xref)

    return np.trapezoid(klout,times)


if __name__=="__main__":

    plot_utils = PlotUtils()
    usebenchmark,_ = metric_params()

    plot_utils.load_du_checkpoint()
    if usebenchmark is not None:
        plot_utils.load_benchmark_solution(usebenchmark)

    #plot_utils.make_contour_plot2(numplots=6)
    plot_utils.make_drift_plot(numplots=5)
    #plot_utils.make_inverted_drift_plot(numplots=5)
    plot_utils.make_sde_plots()
    #plot_utils.checkpoint_plots_du(usebenchmark=usebenchmark)
    #plot_utils.make_cumulant_error_plot_from_csv()
    #plot_utils.make_intermediate_histograms()
    #plot_utils.make_cumulant_error_plot_from_csv()
    