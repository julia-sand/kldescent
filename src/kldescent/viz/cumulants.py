"""Make plots to visualise cumulants"""

import string
import warnings

import matplotlib.pyplot as plt
import numpy as np 

from kldescent.viz.plots import save_fig_, update_mpl

def load_cumulant_csvs(folder):
    """
    Loads time vector, neural network cumulants, and benchmark cumulants from CSV files.

    Parameters
    ----------
    folder : str
        Path to the folder containing the CSV files.

    Returns
    -------
    tvec1 : array-like
        Time vector for cumulants computed by NN.
    tvec2 : array-like
        Time vector for cumulants computed by benchmark.
    nn_cumulants : array-like
        Cumulants from the neural network.
    benchmark_cumulants : array-like
        Benchmark cumulants.
    """

    cumulants_nn = np.loadtxt(open(folder+"/cumulants.csv"),
                    delimiter=",",skiprows=1) 
    cumulants_ref = np.loadtxt(open(folder+"/../ref_cumulants.csv"),
                    delimiter=",",skiprows=1)

    return cumulants_nn[:,0],cumulants_ref[:,0],cumulants_nn[:,1:],cumulants_ref[:,1:]


def make_fig_gridspec(nrows=2,ncols=6):

    #Plotting the cumulants
    fig1 = plt.figure(figsize=(18, 15))#, constrained_layout=True)

    # Create a grid 
    gs = fig1.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1,1,1], height_ratios=[1, 1])
           
    return fig1,gs

def position_mean_free_diffusion(t_vec,q0,p0,epsilon):
    return q0 + epsilon*(1-np.exp(-t_vec))*p0

def momentum_mean_free_diffusion(t_vec,p0):
    return np.exp(-t_vec)*p0

def position_variance_free_diffusion(t_vec,pos_var0,mom_var0,epsilon):
    return pos_var0 + (epsilon**2)*(((1-np.exp(-t_vec))**2)*mom_var0 + 2*t_vec - 3 +4*np.exp(-t_vec) - np.exp(-2*t_vec))

def momentum_variance_free_diffusion(t_vec,mom_var0):
    return 1 - np.exp(-2*t_vec) + np.exp(-2*t_vec)*mom_var0

def cross_correlation_free_diffusion(t_vec,pos_var0,mom_var0,epsilon):
    return (epsilon**2)*np.exp(-t_vec)*(1-np.exp(-t_vec))*mom_var0 + epsilon*(1-2*np.exp(-t_vec)+np.exp(-2*t_vec))

def make_cumulant_plot_label(plot_index):
    return "("+string.ascii_lowercase[plot_index]+")"

def make_cumulant_plot_axes(gs):

    #define plot locations
    axs = [plt.subplot(gs[0, 0:2]),
            plt.subplot(gs[0, 2:4]),
            plt.subplot(gs[0, 4:]),
            plt.subplot(gs[1,:3]),
            plt.subplot(gs[1, 3:])]

    return axs 

def format_cumulant_subplot(ax,index,ylabel_text,t_vec):
    """
    Formats a subplot for cumulant visualization.

    Sets the title, y-axis label, and x-axis limits for a given subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The subplot axes to format.
    index : int
        Index of the cumulant, used to generate the subplot title.
    ylabel_text : str
        Label for the y-axis.
    t_vec : array-like
        Time vector used to define the x-axis range.
    """
    ax.set_title(make_cumulant_plot_label(index),pad = 5,x = 0.07, y = 0.88, zorder = 1000000)
    ax.set_ylabel(ylabel_text)
    ax.set_xlim((-0.1,t_vec[-1]+0.1))
    ax.set_xlabel(r"$t$")
    ax.set_xticks([0,1,2,3,4,5])

def compute_variance_error_bar(cumulant_vec,test_samples):
    return cumulant_vec*np.sqrt(1/test_samples)

def compute_covariance_error_bar(pos_var,mom_var,test_samples):
    return np.sqrt(pos_var*mom_var/test_samples)

def make_cumulant_subplot(ax,#gs_location,
                            ylabel_text,
                            t_vec,
                            cumulants_nn_plot,label,test_samples):
  with warnings.catch_warnings(): #dont print divide by zero warnings (these are handled in the script)
    warnings.simplefilter("ignore")
    
    color = ax.lines[-1].get_color()
    conf_level = 2.58 # corresponds to 99% conf. interval
    if ylabel_text=="Position Mean":
      index = 0 
      ax.plot(t_vec,cumulants_nn_plot[:,0], label=label, zorder=1000,color=color)
      yerr=np.sqrt(cumulants_nn_plot[:,2]/test_samples)
      ax.errorbar(t_vec, cumulants_nn_plot[:,index], conf_level*yerr,alpha=0.8)

    elif ylabel_text=="Momentum Mean":
      index = 1
      ax.plot(t_vec,cumulants_nn_plot[:,1], label=label, zorder=1000,color=color)
      yerr=np.sqrt(cumulants_nn_plot[:,3]/test_samples)
      ax.errorbar(t_vec, cumulants_nn_plot[:,index], conf_level*yerr,alpha=0.8)

    elif ylabel_text=="Position Variance":
      index = 2 
      ax.plot(t_vec,cumulants_nn_plot[:,2], label=label, zorder=1000,color=color)
      yerr=compute_variance_error_bar(cumulants_nn_plot[:,2],test_samples)
      ax.errorbar(t_vec, cumulants_nn_plot[:,2], conf_level*yerr,alpha=0.8)

    elif ylabel_text=="Momentum Variance":
      index = 3 
      ax.plot(t_vec,cumulants_nn_plot[:,3], label=label, zorder=1000,color=color)
      yerr=compute_variance_error_bar(cumulants_nn_plot[:,3],test_samples)
      ax.errorbar(t_vec, cumulants_nn_plot[:,3], conf_level*yerr,alpha=0.8)

    elif ylabel_text=="Cross Correlation":
      index = 4 
      ax.plot(t_vec,cumulants_nn_plot[:,4], label=label, zorder=1000,color=color)
      yerr = compute_covariance_error_bar(cumulants_nn_plot[:,3], cumulants_nn_plot[:,2], cumulants_nn_plot[:,3], test_samples)
      ax.errorbar(t_vec, cumulants_nn_plot[:,4], conf_level*yerr,alpha=0.8)
      
    
    else:
      print(ylabel_text+" not found")

    try:
      #ax.plot(t_vec,cumulants_nn_plot[:,index], label=label, zorder=1000,color=color)
      ax.errorbar(t_vec, cumulants_nn_plot[:,index], yerr,alpha=0.8)
    except: #if mcsamples set to 0, skip the error bars
      pass

    return ax

def plot_free_diffusion_sol(ax,t_vec,cumulant):
      ax.plot(t_vec,cumulant,
                            linestyle="dashed",
                            zorder=0,
                            alpha=0.5,
                            color="gray",
                            label="Free Diffusion")

def add_freediffusion_cumulants_subplot(ax,#gs_location,
                                      ylabel_text,
                                      t_vec,
                                      cumulants_ref_plot,
                                      epsilon):
      
    if ylabel_text=="Position Mean":
        index = 0 
        plot_free_diffusion_sol(ax,t_vec,position_mean_free_diffusion(t_vec,cumulants_ref_plot[0,index],cumulants_ref_plot[0,1],epsilon))
        ax.set_ylim((-0.2,0.2))

    elif ylabel_text=="Momentum Mean":
        index = 1
        plot_free_diffusion_sol(ax,t_vec,momentum_mean_free_diffusion(t_vec,cumulants_ref_plot[0,index]))
        ax.set_ylim((-0.2,0.2))

    elif ylabel_text=="Position Variance":
        index = 2 
        plot_free_diffusion_sol(ax,t_vec,position_variance_free_diffusion(t_vec,cumulants_ref_plot[0,index],cumulants_ref_plot[0,index+1],epsilon))
    
    elif ylabel_text=="Momentum Variance":
        index = 3 
        plot_free_diffusion_sol(ax,t_vec,momentum_variance_free_diffusion(t_vec,cumulants_ref_plot[0,index]))
    elif ylabel_text=="Cross Correlation":
        index = 4 
        plot_free_diffusion_sol(ax,t_vec,cross_correlation_free_diffusion(t_vec,cumulants_ref_plot[0,2],cumulants_ref_plot[0,3],epsilon))
       
def plot_cumulants(t_vec1,t_vec2,cumulants,labels,test_samples,
                      benchmark=None,
                      add_free_diffusion=False,
                      epsilon=None):
    """
    Plots cumulants computed from dynamical evolutions.

    The function generates a multi-panel figure comparing cumulants computed
    from two time vectors and multiple datasets. It optionally overlays
    benchmark (perturbative) predictions and free diffusion curves.

    Parameters
    ----------
    t_vec1 : array-like
        First time vector for cumulant evaluation.
    t_vec2 : array-like
        Second time vector for cumulants evaluation by the benchmark.
    cumulants : list
        List of cumulant arrays to plot. Each entry should correspond to a
        dataset matching the provided time vectors.
    labels : list of str
        Legend labels corresponding to each cumulant dataset.
    test_samples : int
        Number of samples used to estimate cumulants and compute error bars.
    benchmark : object, optional
        Object providing perturbative predictions via
        ``get_perturbative_cumulant(index)``. If None, no benchmark is plotted.
    add_free_diffusion : bool, optional
        If True, overlays free diffusion prediction curves.
    epsilon : float, optional
        Parameter required for free diffusion curves.
        Must be provided if ``add_free_diffusion=True``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing all cumulant subplots and legend.
    """

    fig1,gs_cumulants = make_fig_gridspec()
    
    #cumulant labels
    cumulant_names = ["Position Mean",
                      "Position Variance",
                      "Momentum Mean",
                      "Momentum Variance",
                      "Cross Correlation"]
    axs = make_cumulant_plot_axes(gs_cumulants)

    for ax, cumulant in zip(axs,enumerate(cumulant_names)):

        for t_vec,cumulant_vec,legend_label in zip([t_vec1,t_vec2],cumulants,labels):

            make_cumulant_subplot(ax,cumulant[1],
                                          t_vec,cumulant_vec,legend_label,
                                          test_samples)
      
        if add_free_diffusion:
            assert epsilon is not None, "No value of epsilon given. Plots interrupted." #check that epsilon is passed correctly
            add_freediffusion_cumulants_subplot(ax,cumulant[1],t_vec1,cumulants[0],epsilon)

        if benchmark is not None:
            try: 
                t_vectemp, cumulant_temp = benchmark.get_perturbative_cumulant(cumulant[1])
                ax.plot(t_vectemp, cumulant_temp,label="Benchmark")
            except FileNotFoundError:
                pass 

        format_cumulant_subplot(ax,cumulant[0],cumulant[1],t_vec1)
    
    plt.tight_layout()

    fig1.subplots_adjust(bottom=0.15)
    h,l = axs[0].get_legend_handles_labels()
    legend = fig1.legend(h,l, 
                            frameon = False,
                            handlelength = 1,
                            ncols = 5,
                            columnspacing=0.4)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))
    

    return fig1

if __name__=="__main__":
  from kldescent.utils.parser import fetch_results_folder_from_cmd,checkpoint_plot_params

  update_mpl()

  folder_base = fetch_results_folder_from_cmd()
  
  params = checkpoint_plot_params()
  epoch = params["epoch"]
  folder = folder_base+f"/plots_{epoch}"

  t_vec1,t_vec2,cumulants_nn,cumulants_ref = load_cumulant_csvs(folder)

  #cumulant figs and csv 
  fig = plot_cumulants(t_vec1,t_vec2,[cumulants_nn,cumulants_ref],
                                        ["NN estimated","Benchmark"],
                                        0,
                                        benchmark=None,
                                        add_free_diffusion=True,
                                        epsilon=0.2)

  save_fig_(fig,"cumulants1",loc=folder)
  #save_cumulants_csv(t_vec1,cumulants_nn,loc=self.resultsdir)
  #save_cumulants_csv(t_vec2,cumulants_ref,loc=self.resultsdir,cumulants_filename="ref_cumulants")
  plt.close("all")
  