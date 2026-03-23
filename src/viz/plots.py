import os

import numpy as np 
from sklearn.neighbors import KernelDensity

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes

'''
This file holds all the functions used in plotting.
'''

def save_cumulants_csv(t_vec,cumulants_nn,loc,cumulants_filename="cumulants"):

  a = np.concatenate((t_vec[...,np.newaxis],cumulants_nn),axis=1)
  #np.savetxt(self.subdir+"/"+cumulants_filename+".csv", a, delimiter=",")
  np.savetxt(loc+"/"+cumulants_filename+".csv", a ,fmt="%10.6f",header = "t,posmean,mommean,posvar,momvar,xcorr",delimiter=',')
  return 

def get_inset_axes(ax):
  return [c for c in ax.get_children() if isinstance(c, Axes)]
    

def make_fig_subplots(nrows,ncols,squeeze=True):

  fig,axs = plt.subplots(ncols=ncols,nrows=nrows,squeeze=squeeze)
  fig.set_figheight(15)
  fig.set_figwidth(25)

  return fig,axs

def make_legend(fig,handles,labels): 
  
  fig.legend(handles,labels,
                      frameon = False,
                      handlelength = 1,
                      ncols = len(labels),
                      bbox_to_anchor=(1,0),
                      bbox_transform=fig.transFigure)

def save_fig_(fig,filename,loc):
  fig.savefig(loc+"/"+filename+".png", bbox_inches='tight')
  #fig.savefig(self.subdir+"/"+filename+".pdf",format="eps", bbox_inches='tight')
  #fig.savefig(self.subdir+"/"+filename+".eps",format="pdf", bbox_inches='tight')
  plt.close()

def normal_distribution(x,mu=0,sigmasq=1):
  
  return np.exp(-((x-mu)**2)/(2*sigmasq))/np.sqrt(2*sigmasq*np.pi)
  
def get_inset_handles_labels(ax): 
    """Find the handles and labels for data plotted on inset axes of ax and returns the 
    unique labels that can be used in a legend. """
    
    #fetch all inset axes for ax
    axes_temp = get_inset_axes(ax)

    #return all labels and handles for each inset ax as dictionary
    labels_handles = {}
    
    for c in axes_temp:
      handles,labels = c.get_legend_handles_labels() 
      ltemp = {label: handle for handle, label in zip(handles,labels)}
      labels_handles.update(ltemp)
      
    return labels_handles #labels_handles.values(),labels_handles.keys()

def get_all_subplot_handles_labels(fig):

  d0 = {}
  for ax in fig.axes:
    dtemp = get_inset_handles_labels(ax) #returns dictionary of labels and handles
    d0.update(dtemp)

  return d0.values(),d0.keys() 

def update_mpl():
  """
  Sets default parameters for any matplotlib plotting formatting
  """

  mpl.use('Agg')  # Use the 'Agg' backend for non-GUI rendering

  #update matplotlib config
  mpl.rcParams['lines.linewidth'] = 3
  mpl.rcParams['xtick.labelsize'] = 20
  mpl.rcParams['ytick.labelsize'] = 20
  mpl.rcParams['font.size'] = 22
  mpl.rcParams['axes.labelsize'] = 22

class PlotFuns():
  def __init__(self,boundary_data=None):

    update_mpl()

    self.boundary = boundary_data
    self.folder = os.path.dirname(__file__)
    
    #borders of xaxis
    self.qmax = 8
    
    #colors
    self.c1 = "#1f78b4" #darkblue
    self.c2 = "#a6cee3" #lightblue
    self.c3 = "#33a02c" #dark green
    self.c4 = "#b2df8a" #light green
 
    
    self.mut_p = 0
    self.sigsqt_p = 1
    self.mut_q = 0
    self.sigsqt_q = 1

    self.subdir = self.folder #default to folder
  
  def setresultsdir(self,folder):
    #set output location 
    self.folder = folder
    self.subdir = folder #also set the subdirectory to default to folder name

  def setsubdir(self, subdir):
    self.subdir = self.folder+subdir #os.path.join(self.folder,subdir)

  def format_drift_ax(self,ax1,time):
    
    ax1.set_title(f"t={time:.4f}")
    ax1.set_xlabel("q")
    #ax1.set_ylim((-1.8,1.8))

  def format_distribution_ax(self,ax1):
    ax1.set_xlim((-self.qmax,self.qmax))
    ax1.set_xlabel("q")
    #ax1.set_ylim((-0.01,0.65))
    ax1.set_ylim(bottom=-0.01)
    #ax1.set_ylabel(r"$\rho$")

  def plot_reference(self,ax1,x,du):

    ax1.plot(x,du,
            label="Reference")

  #@profile
  def plot_drift(self,times,net_vals,benchmark_vals):

    fig,ax = make_fig_subplots(nrows=1,ncols=len(times),squeeze=True)
    #ax = plt.subplots()
  
    for t in range(len(times)):  
      
      ax[t].plot(np.linspace(-self.qmax,self.qmax,1000),
                net_vals[t],
                label="N.N.")
      
      self.format_drift_ax(ax[t],times[t])
    
      ax[t].set_ylabel(r"$\mathbb{U}$")
      #ax[t].set_ylim((-5,5))
      
      if benchmark_vals is not None:
        
        ax[t].plot(np.linspace(-self.qmax,self.qmax,1000),
            benchmark_vals[t],
            label="Benchmark N.N.")
      
      else:
        pass
        #print("No reference solution was given. Plots will not include reference values.")
    
    h,l = ax[0].get_legend_handles_labels()
    #plt.legend()
    legend = fig.legend(h,l,
              frameon = False,
              handlelength = 1,
              ncols = 2)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig

  def fit_kde(self,xs):

    kde = KernelDensity(kernel = "epanechnikov",bandwidth=0.1).fit(xs[np.isfinite(xs)].reshape(-1,1))
    return kde.score_samples(np.linspace(-self.qmax,self.qmax,num =1000).reshape(-1,1))

  ##scatter plot of final distribution 
  def marginal_distribution(self,ax,xs,plottime=None,benchmark=None):
    """
    Function that plots the an estimate of the marginal densities
    computed by sampled points

    args:
      xs : ndarray
        sampled positions for the position and momentum
      plottime : float
        time at which the samples were taken. Defaults to final time from boundary
    """
    
    if plottime is None:
      plottime = self.boundary.params["Tf"]
    
    ax_pos = ax.inset_axes([0,0,0.48,1],sharey=ax)
    ax_mom = ax.inset_axes([0.52,0,0.48,1],sharey=ax)#divider.append_axes("left", size="50%", pad=0.1, sharey=ax)
    
    #remove ax xaxis
    ax.spines.bottom.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticks([])

    #add titles
    ax.set_title(f"t = {plottime}")
    ax_pos.set_ylabel("Position Marginal")
    ax_mom.set_ylabel("Momentum Marginal")
    
    #set plot limits
    ax_mom.set_xlim((-self.qmax,self.qmax))
    ax_pos.set_xlim((-self.qmax,self.qmax))
    
    #add perturbative analytic predictions distributions to plots if available
    if benchmark is not None:
      try:
        xref,posref = benchmark.get_quantity(plottime,"underdamped marginal", data="perturbative underdamped",time_type="t0")   
        ax_pos.plot(xref,posref,color="green",alpha=0.5,label="Benchmark Prediction")
        label2 = "Benchmark Evolved"
      except ValueError: 
        print(f"Benchmark data not available for t = {plottime}")
        label2 = "Free Diffusion"
    else:
      label2 = "Free Diffusion"
    #set iteration properties
    colors = [self.c1,"orange"]
    labels = ["Controlled by NN.",label2]

    for x in enumerate(xs):
      
      #plot kde fitted distributions
      ax_mom.plot(np.linspace(-self.qmax,self.qmax,num=1000), np.exp(self.fit_kde(x[1][:,1])), color=colors[x[0]],label=labels[x[0]])
      ax_pos.plot(np.linspace(-self.qmax,self.qmax,num=1000), np.exp(self.fit_kde(x[1][:,0])), color=colors[x[0]],label=labels[x[0]]) 
    
    #add target boundary distributions to plots
    ax_pos.fill_between(np.linspace(-self.qmax,self.qmax,num=1000),
            self.boundary.q_marginal_final(np.linspace(-self.qmax,self.qmax,num=1000)),
            color="gray",alpha=0.1,label="target")
    ax_mom.fill_between(np.linspace(-self.qmax,self.qmax,num=1000),
        normal_distribution(np.linspace(-self.qmax,self.qmax,num=1000),mu=0,sigmasq=1),
                    color="gray",alpha=0.1,label="target")
    
    #add formatting
    self.format_distribution_ax(ax_mom)
    self.format_distribution_ax(ax_pos)

    #move the yaxis labels to opposite side for mom plot
    ax_mom.yaxis.tick_right()
    ax_mom.yaxis.set_label_position("right")
    
    #set plot limits
    ax_mom.set_xlim((-5,5))
    ax_pos.set_xlim((-3,3))
    
    return 
  
  def make_marginal_distribution_fig(self,xvals,plottime,benchmark=None):

    #make figure object
    fig = plt.figure(figsize=(15,8))
    ax0 = plt.subplot(1,1,1)

    self.marginal_distribution(ax0,xvals,plottime,benchmark)
    
    h,l = get_all_subplot_handles_labels(fig)

    #make legend 
    legend = fig.legend(h,l,ncols= 4, 
                            frameon=False,handlelength = 1)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))
    
    fig.tight_layout()
    
    return fig  
    


  def update_final_cumulants(self,cumulants_nn):
    self.mut_p = cumulants_nn[1]
    self.sigsqt_p = cumulants_nn[3]
    self.mut_q = cumulants_nn[0]
    self.sigsqt_q = cumulants_nn[2]
     

if __name__=="__main__":
  
  update_mpl()
