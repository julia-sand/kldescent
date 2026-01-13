import os
import string

import numpy as np 
from sklearn.neighbors import KernelDensity
import torch

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

    self.titlepad = 5
    self.titlex = 0.07
    self.titley = 0.88
    
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

  def make_plot_label(self,plot_index):
    return "("+string.ascii_lowercase[plot_index]+")"

  def plot_marginal_joint(self,ax,x0_marginal,x0_init):

    ax.plot(x0_init,x0_marginal
                      ,color=self.c1,label="Girsanov")

  def get_norm(self,x,y):
    idx = np.argsort(x)
    return np.trapz(y[idx],x[idx])

  def add_boundary_conditions_scatter_panel(self,ax_pmarginal,ax_qmarginal,time):
    
    if time==0:
      ax_qmarginal.fill_between(
                self.boundary.q_marginal_initial(np.linspace(-self.qmax,self.qmax,)),np.linspace(-self.qmax,self.qmax,),color="orange",label="Assigned Initial Distribution")
      
      ax_pmarginal.fill_between(np.linspace(-self.qmax,self.qmax,),
                self.normal_distribution(np.linspace(-self.qmax,self.qmax,),0,1),color="orange",label="Assigned Initial Distribution")
    
    elif time==self.boundary.params["Tf"]:
      ax_qmarginal.fill_between(
                self.boundary.q_marginal_final(np.linspace(-self.qmax,self.qmax,)),np.linspace(-self.qmax,self.qmax,),color="orange",label="Target Final Distribution")
      
      ax_pmarginal.fill_between(np.linspace(-self.qmax,self.qmax,),
                self.normal_distribution(np.linspace(-self.qmax,self.qmax,),0,1),color="darkorange",label="Target Final Distribution")
    
    else: 
      pass


  def scatter_panel_layout(self,ax,time=None):

    if time == None:
      time = self.boundary.params["Tf"]

    if self.boundary is not None:  
      plot_title_value = round(time/self.boundary.params["Tf"],2)
      ax.set_title(f"t={plot_title_value}Tf")
    else: 
      ax.set_title(f"t={time}")

    #make inset axes to show marginal distributions
    ax_pmarginal = ax.inset_axes([0, 1.05, 1, 0.6])
    ax_qmarginal = ax.inset_axes([-0.65, 0, 0.6, 1])
    ax_qmarginal.invert_xaxis()

    #set x axis limits
    ax.set_ylim((-self.qmax,self.qmax))
    ax.set_xlim((-self.qmax,self.qmax))
    ax_qmarginal.set_ylim((-self.qmax,self.qmax))
    ax_pmarginal.set_xlim((-self.qmax,self.qmax))

    ###ticks formatting and labelling
    ax_pmarginal.yaxis.tick_right()
    ax.yaxis.tick_right()
    
    ax_pmarginal.tick_params(
        axis='y',
        which='both',
        length=5,
        rotation=-20,
        pad = -3)

    ax_qmarginal.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',
        rotation=70,
        length =5,
        pad = -3,
        labelleft=False)
    
    ax_qmarginal.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      #both major and minor ticks
        left=False,
        labelleft=False)

    ax_pmarginal.set_xticklabels([])
   
    ax.set_xlabel(r"$p$")
    ax_qmarginal.set_xlabel(r"$\rho_t(q)$")
    ax_qmarginal.set_xlim(right=-0.01,left=0.6)

    ax.set_ylabel(r"$q$")
    ax_pmarginal.set_ylabel(r"$\rho_t(p)$")
    ax.yaxis.set_label_position("right")
    ax_pmarginal.set_ylim(bottom=-0.01,top=0.6)

    ax.set_aspect('equal', adjustable='box')
    #ax.set_xticks([-2.5,0,2.5])
    
    if self.boundary is not None:
      self.add_boundary_conditions_scatter_panel(ax_pmarginal,ax_qmarginal,time)

    return ax, ax_pmarginal,ax_qmarginal

  def joint_distributions_scatter_panel(self,
                                        ax,
                                        ax_pmarginal,
                                        ax_qmarginal,
                                        joint_out,
                                        xs,time,vmax=0.5,
                                        refsol=None):
    """
    Plots one panel of the joint distribution, where the 
    center shows the joint and marginals are estimated

    args:
      ax: artist
        axes where to plot. can be a gridspec subplot
      joint_out: arr
        joint distribution (n**2,1) as a numpy array
      xs: arr
        2d input data shaped (n**2,2) as numpy arr
      vmax: float
        maximum value for the scatter plots color scale
    """

    #pdb.set_trace()
    n = joint_out.shape[0]
    
    #pdb.set_trace()
    joint_out_temp = np.nan_to_num(joint_out.reshape((int(np.sqrt(n)),int(np.sqrt(n)))),nan=0,posinf=0,neginf=0)
    p_init = np.unique(xs) #np.sort(Q[[0],:].flatten()) #one axis 
    dx = np.abs(p_init[1]-p_init[0])

    #compute marginals
    qmarg = np.trapz(joint_out_temp,x = p_init,axis=0)#np.sum(joint_out_temp,axis=0) #np.trapz(joint_out_temp,x = p_init,axis=0)
    pmarg = np.trapz(joint_out_temp,x = p_init,axis=1)
    qnorm = self.get_norm(p_init, qmarg)
    pnorm = self.get_norm(p_init, pmarg)

    #plot and format marginals
    self.plot_marginal_joint(ax_qmarginal,p_init,qmarg/qnorm)
    self.plot_marginal_joint(ax_pmarginal,pmarg/pnorm,p_init)

    if refsol is not None:
      #plot marginals
      x,ptx = refsol.get_quantity(np.round(time,6),quantity = "underdamped marginal",data="perturbative underdamped")
      ax_qmarginal.plot(ptx,x,color="gray",linestyle="dashed",label="Perturbative Calculated")

    #plot joint distribution
    ax.scatter(xs[...,0]/(qnorm*pnorm),xs[...,1]/(qnorm*pnorm),c=joint_out_temp.T.flatten()
              ,cmap="Blues",norm="log",vmin=0.0001,vmax=vmax)
    
  def make_colourbar(self,fig,vmax):
    #adds a colour

    sm =  ScalarMappable(norm="log",cmap=plt.get_cmap("Blues"))
    sm.set_array([])
    sm.set_clim(vmin=0.001, vmax=vmax)

    cbar_ax = fig.add_axes([0.2, -0.15, 0.55, 0.05])
    cb = fig.colorbar(sm,
                      cax=cbar_ax,
                      orientation="horizontal")

    cb.ax.set_xlabel("Joint Distribution")

  def make_legend(self,fig,handles,labels): 
    
    fig.legend(handles,labels,
                        frameon = False,
                        handlelength = 1,
                        ncols = len(labels),
                        bbox_to_anchor=(1,0),
                        bbox_transform=fig.transFigure)

  def format_drift_ax(self,ax1,time):
    ax1.set_xlim((-10,10))#self.qmax,self.qmax))
    #ax1.set_xlim((-2,2))
    ax1.set_title(f"t={time}")
    ax1.set_xlabel("q")
    #ax1.set_ylim((-1.8,1.8))

  def normal_distribution(self,x,mu=0,sigmasq=1):
    
    return np.exp(-((x-mu)**2)/(2*sigmasq))/np.sqrt(2*sigmasq*np.pi)
      
  def format_distribution_ax(self,ax1):
    ax1.set_xlim((-self.qmax,self.qmax))
    ax1.set_xlabel("q")
    ax1.set_ylim((-0.01,0.65))
    #ax1.set_ylabel(r"$\rho$")

  def plot_reference(self,ax1,x,du):

    ax1.plot(x,du,
            label="Reference")

  #@profile
  def plot_drift(self,times,net,refsol=None):

    fig,ax = self.make_fig_subplots(nrows=1,ncols=len(times),squeeze=True)#plt.subplots()#plt.figure(figsize=(15,8))
    #ax = plt.subplots()
  
    for t in range(len(times)):  
      
      ax[t].plot(np.linspace(-self.qmax,self.qmax,1000),
              self.evaluate_net_output(net,times[t]),
                label="N.N.")
      
      self.format_drift_ax(ax[t],times[t])
    
      ax[t].set_ylabel(r"$\mathbb{U}$")
      ax[t].set_ylim((-5,5))
      if refsol is not None:
        #xref,duref = refsol.get_quantity(round(times[t],3), "underdamped drift", data="perturbative underdamped")
        #ax[t].plot(xref,-0.2*duref/(2**0.5),label="Perturbative")
        ax[t].plot(np.linspace(-self.qmax,self.qmax,1000),
                self.evaluate_net_output(refsol.duref_net,times[t]),
                label="Perturbative N.N.")
      else:
        print("No reference solution was found. Plots will not include reference values.")
    
    plt.tight_layout()

    h,l = ax[0].get_legend_handles_labels()
    #plt.legend()
    legend = fig.legend(h,l,
              frameon = False,
              handlelength = 1,
              ncols = 2)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))
    self.save_fig_(fig,"drift")
    plt.close()

  def fit_kde(self,xs):

    kde = KernelDensity(kernel = "epanechnikov",bandwidth=0.1).fit(xs[np.isfinite(xs)].reshape(-1,1))
    return kde.score_samples(np.linspace(-self.qmax,self.qmax,num =1000).reshape(-1,1))

  def make_fig_subplots(self,nrows,ncols,squeeze=True):

    fig,axs = plt.subplots(ncols=ncols,nrows=nrows,squeeze=squeeze)
    fig.set_figheight(15)
    fig.set_figwidth(25)

    return fig,axs

  ##scatter plot of final distribution 
  def marginal_distribution(self,ax,xs,plottime=None,refsol=None):
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
    
    #set iteration properties
    colors = ["orange",self.c1]
    labels = ["Controlled by NN.","Perturbative"]

    for x in enumerate(xs):
      
      #plot kde fitted distributions
      ax_mom.plot(np.linspace(-self.qmax,self.qmax,num=1000), np.exp(self.fit_kde(x[1][:,1])), color=colors[x[0]],label=labels[x[0]])
      ax_pos.plot(np.linspace(-self.qmax,self.qmax,num=1000), np.exp(self.fit_kde(x[1][:,0])), color=colors[x[0]],label=labels[x[0]]) 
    
    #add perturbative  distributions to plots if available
    if refsol is not None:
         
      xref,posref = refsol.get_quantity(plottime,"underdamped marginal", data="perturbative underdamped")   
      ax_pos.fill_between(xref,posref,color="green",alpha=0.5,zorder=1000,label="Perturbative Computed")
    
    #add reference distributions to plots
    ax_pos.fill_between(np.linspace(-self.qmax,self.qmax,num=1000),
            self.boundary.q_marginal_final(np.linspace(-self.qmax,self.qmax,num=1000)),
            color="gray",alpha=0.1,label="target")
    ax_mom.fill_between(np.linspace(-self.qmax,self.qmax,num=1000),
        self.normal_distribution(np.linspace(-self.qmax,self.qmax,num=1000),mu=0,sigmasq=1),
                    color="gray",alpha=0.1,label="target")
    
    #add formatting
    self.format_distribution_ax(ax_mom)
    self.format_distribution_ax(ax_pos)

    #move the yaxis labels to opposite side for mom plot
    ax_mom.yaxis.tick_right()
    ax_mom.yaxis.set_label_position("right")
    
    h,l = ax_pos.get_legend_handles_labels()
    return h,l
  
  def make_marginal_distribution_fig(self,xvals,plottime):

    #make figure object
    fig = plt.figure(figsize=(15,8))#self.make_fig_subplots(nrows=1,ncols=1)#plt.figure(figsize=(15,8))
    ax0 = plt.subplot(1,1,1)

    h,l = self.marginal_distribution(ax0,xvals,plottime)
    
    #make legend 
    legend = fig.legend(h,l,ncols= 4, 
                            frameon=False,handlelength = 1)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))
    
    fig.tight_layout()
    
    self.save_fig_(fig,f"/marginals_Time_{plottime}")#.png")
    plt.close()

  def get_loss_output(self,filename=None):
    if filename is not None:
      arr = np.loadtxt(filename,
                          delimiter=",",skiprows=1)  
    else:
      ##fetch the logged results for current run
      arr = np.loadtxt(self.folder+"/loss.csv",
                          delimiter=",",skiprows=1)  

    return arr

  def plot_lhat(self,ax,arr,window_size=5):

    ax.plot(arr[:,0],arr[:,1],label=r"$\hat\mathcal{L}$")
    ax.plot(np.squeeze(arr[:,0])[window_size:-window_size],self.movingaverage(np.squeeze(arr[:,1]),window_size),linestyle="dashed")

    #set axis labels
    ax.set_xlabel("epochs")
    ax.set_ylabel(r"Effective Loss, $\hat\mathcal{L}$")
    
    return     

  def plot_dynamic_kl(self,ax,arr,window_size=5):

    #arr = self.get_loss_output()

    ax.plot(arr[:,0],arr[:,2],label=r"K.L. Divergence")
    ax.plot(np.squeeze(arr[:,0])[window_size:-window_size],self.movingaverage(np.squeeze(arr[:,2]),window_size),linestyle="dashed")

    #set axis labels
    ax.set_xlabel("epochs")
    ax.set_ylabel(r"(dynamic) K.L.")

  def plot_static_kl(self,ax,arr,window_size=5):

    #arr = self.get_loss_output()

    ax.plot(arr[:,0],arr[:,3],label=r"K.L. Divergence")
    ax.plot(np.squeeze(arr[:,0])[window_size:-window_size],self.movingaverage(np.squeeze(arr[:,3]),5),linestyle="dashed")

    #set axis labels
    ax.set_xlabel("epochs")
    ax.set_ylabel(r"(static) K.L.")

  def plot_learningrate(self,ax,arr):


    ax.plot(arr[:,0],-arr[:,-1]*10,label=r"$\eta$")
    
    return     

  def plot_loss_metrics(self,filename=None):

    fig = plt.figure(figsize=(15,8))

    arr = self.get_loss_output(filename)

    ax1 = fig.subplot(131)
    ax2 = fig.subplot(132)
    ax3 = fig.subplot(133)
    try:
      self.plot_lhat(ax1,arr)
      self.plot_dynamic_kl(ax2,arr)
      self.plot_static_kl(ax3,arr)
      
      plt.tight_layout()
      self.save_fig_(fig,"losses")
      
      plt.close()
    
    except IndexError:
      print("Not enough data to plot the loss metrics.\n \
               Ensure that there has been at least one epoch in the current run and logging frequency is configured correctly.")
      plt.close()
    
    return     

  ##@profile
  def evaluate_net_output(self,net,time):
    with torch.no_grad():
        #computes neural network output (forward pass) to use in plotting
      return net(torch.column_stack((time*torch.ones((1000,1),requires_grad=False,dtype=torch.float32)
                                                    ,torch.linspace(-self.qmax,self.qmax,1000,requires_grad=False)))).detach()
               
  def position_mean_free_diffusion(self,t_vec,q0,p0):
    return q0 + self.boundary.params["epsilon"]*(1-np.exp(-t_vec))*p0

  def momentum_mean_free_diffusion(self,t_vec,p0):
    return np.exp(-t_vec)*p0

  def position_variance_free_diffusion(self,t_vec,pos_var0,mom_var0):
    return pos_var0 + (self.boundary.params["epsilon"]**2)*(((1-np.exp(-t_vec))**2)*mom_var0 + 2*t_vec - 3 +4*np.exp(-t_vec) - np.exp(-2*t_vec))

  def momentum_variance_free_diffusion(self,t_vec,mom_var0):
    return 1 - np.exp(-2*t_vec) + np.exp(-2*t_vec)*mom_var0

  def cross_correlation_free_diffusion(self,t_vec,pos_var0,mom_var0):
    return (self.boundary.params["epsilon"]**2)*np.exp(-t_vec)*(1-np.exp(-t_vec))*mom_var0 + self.boundary.params["epsilon"]*(1-2*np.exp(-t_vec)+np.exp(-2*t_vec))

  def save_cumulants_csv(self,t_vec,cumulants_nn,cumulants_filename="cumulants"):

    a = np.concatenate((t_vec[...,np.newaxis],cumulants_nn),axis=1)
    np.savetxt(self.subdir+"/"+cumulants_filename+".csv", a, delimiter=",")
    return 

  def plot_free_diffusion_sol(self,ax,t_vec,cumulant):
      ax.plot(t_vec,cumulant,
                            linestyle="dashed",
                            zorder=0,
                            alpha=0.5,
                            color="gray",
                            label="Free Diffusion")

  def add_freediffusion_cumulants_subplot(self,ax,#gs_location,
                                      ylabel_text,t_vec,cumulants_ref_plot):
      
    if ylabel_text=="Position Mean":
      index = 0 
      self.plot_free_diffusion_sol(ax,t_vec,self.position_mean_free_diffusion(t_vec,cumulants_ref_plot[0,index],cumulants_ref_plot[0,1]))

    elif ylabel_text=="Momentum Mean":
      index = 1
      self.plot_free_diffusion_sol(ax,t_vec,self.momentum_mean_free_diffusion(t_vec,cumulants_ref_plot[0,index]))

    elif ylabel_text=="Position Variance":
      index = 2 
      self.plot_free_diffusion_sol(ax,t_vec,self.position_variance_free_diffusion(t_vec,cumulants_ref_plot[0,index],cumulants_ref_plot[0,index+1]))
    elif ylabel_text=="Momentum Variance":
      index = 3 
      self.plot_free_diffusion_sol(ax,t_vec,self.momentum_variance_free_diffusion(t_vec,cumulants_ref_plot[0,index]))
    elif ylabel_text=="Cross Correlation":
      index = 4 
      self.plot_free_diffusion_sol(ax,t_vec,self.cross_correlation_free_diffusion(t_vec,cumulants_ref_plot[0,2],cumulants_ref_plot[0,3]))
      
    
  def make_cumulant_subplot(self,ax,#gs_location,
                            ylabel_text,
                            t_vec,
                            cumulants_nn_plot,label,test_samples):
        
    if ylabel_text=="Position Mean":
      index = 0 
      yerr=np.sqrt(cumulants_nn_plot[:,2]/test_samples)

    elif ylabel_text=="Momentum Mean":
      index = 1
      yerr=np.sqrt(cumulants_nn_plot[:,3]/test_samples)

    elif ylabel_text=="Position Variance":
      index = 2 
      yerr=cumulants_nn_plot[:,index]*np.sqrt(2/(test_samples-1))

    elif ylabel_text=="Momentum Variance":
      index = 3 
      yerr=cumulants_nn_plot[:,index]*np.sqrt(2/(test_samples-1))
      
    
    elif ylabel_text=="Cross Correlation":
      index = 4 
      yerr = np.sqrt((1-cumulants_nn_plot[:,index]**2)/(test_samples-2))
    
    ax.errorbar(t_vec, cumulants_nn_plot[:,index], yerr,alpha=0.8)
    
    color = ax.lines[-1].get_color()
    ax.plot(t_vec,cumulants_nn_plot[:,index], label=label, zorder=1000,color=color)

  def format_cumulant_subplot(self,ax,index,ylabel_text,t_vec):
    ax.set_title(self.make_plot_label(index),pad = self.titlepad,x = self.titlex, y = self.titley,zorder = 1000000)
    ax.set_ylabel(ylabel_text)
    ax.set_xlim((-0.1,t_vec[-1]+0.1))

  def make_fig_gridspec(self,nrows=2,ncols=6):

    # Plotting the cumulants
    fig1 = plt.figure(figsize=(15, 8))#, constrained_layout=True)

    # Create a grid 
    gs = fig1.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1,1,1], height_ratios=[1, 1])
           
    return fig1,gs

  def plot_cumulants(self,t_vec,cumulants,labels,test_samples):
    """
    Plots the cumulants which are computed by the evolutions
    
    args: 
      t_vec : ndarray
        array of times at which cumulants are evaluated
      cumulants : list
        list of arrays which are plotted iteratively
      labels : list
        list of labels to show in the legend for each of the cumulants plotted
      test_samples : int
        number of samples used for computing estimates of the cumulants. Computes error bar estimates 

    returns:
      fig : figure
        matplotlib figure with cumulants plotted and legend 
    """

    fig1,gs_cumulants = self.make_fig_gridspec()

    for cumulant_vec,legend_label in zip(cumulants,labels):
               
        self.make_cumulant_subplot(plt.subplot(gs_cumulants[0, 0:2]),"Position Mean",
                                          t_vec,cumulant_vec,legend_label,
                                          test_samples)
        self.add_freediffusion_cumulants_subplot(plt.subplot(gs_cumulants[0, 0:2]),"Position Mean",t_vec,cumulant_vec) 

        self.make_cumulant_subplot(plt.subplot(gs_cumulants[0, 2:4]),"Position Variance",t_vec,cumulant_vec,legend_label,test_samples)
        self.add_freediffusion_cumulants_subplot(plt.subplot(gs_cumulants[0, 2:4]),"Position Variance",t_vec,cumulant_vec) 

        self.make_cumulant_subplot(plt.subplot(gs_cumulants[0, 4:]),"Momentum Mean",t_vec,cumulant_vec,legend_label,test_samples)
        self.add_freediffusion_cumulants_subplot(plt.subplot(gs_cumulants[0, 4:]),"Momentum Mean",t_vec,cumulant_vec) 

        self.make_cumulant_subplot(plt.subplot(gs_cumulants[1,:3]),"Momentum Variance",t_vec,cumulant_vec,legend_label,test_samples)
        self.add_freediffusion_cumulants_subplot(plt.subplot(gs_cumulants[1,:3]),"Momentum Variance",t_vec,cumulant_vec) 

        self.make_cumulant_subplot(plt.subplot(gs_cumulants[1, 3:]),"Cross Correlation",t_vec,cumulant_vec,legend_label,test_samples)
        self.add_freediffusion_cumulants_subplot(plt.subplot(gs_cumulants[1, 3:]),"Cross Correlation",t_vec,cumulant_vec)                      
              
    #add formatting to each subplot
    for ax,ylabel_text in zip(enumerate(fig1.get_axes()),["Position Mean","Position Variance","Momentum Mean","Momentum Variance","Cross Correlation"]):
      self.format_cumulant_subplot(ax[1],ax[0],ylabel_text,t_vec)
    
    plt.tight_layout()

    h,l = ax[1].get_legend_handles_labels()
    legend = fig1.legend(h,l, 
                            frameon = False,
                            handlelength = 1,
                            ncols = 5,
                            columnspacing=0.4)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))
    
    return fig1

  def save_fig_(self,fig,filename):
    fig.savefig(self.subdir+"/"+filename+".png", bbox_inches='tight')
    #fig.savefig(self.subdir+"/"+filename+".pdf",format="eps", bbox_inches='tight')
    #fig.savefig(self.subdir+"/"+filename+".eps",format="pdf", bbox_inches='tight')
    plt.close()

  def update_final_cumulants(self,cumulants_nn):
    self.mut_p = cumulants_nn[1]
    self.sigsqt_p = cumulants_nn[3]
    self.mut_q = cumulants_nn[0]
    self.sigsqt_q = cumulants_nn[2]
     
  def movingaverage(self, data, window_size):
      data_means = np.convolve(data, np.ones(int(window_size))/float(window_size), 'same')
      return data_means[window_size:-window_size]

  def get_all_subplot_handles_labels(self,fig):

    d0 = {}
    for ax in fig.axes:
      dtemp = self.get_inset_handles_labels(ax) #returns dictionary of labels and handles
      d0.update(dtemp)

    return d0.values(),d0.keys()


  def get_inset_handles_labels(self,ax): 
      """Finds the handles and labels for data plotted on inset axes of ax and returns the 
      unique labels that can be used in a legend. 
      """
      
      #fetch all inset axes for ax
      axes_temp = [c for c in ax.get_children() if isinstance(c, Axes)]
           
      #return all labels and handles for each inset ax as dictionary
      labels_handles = {}
      
      for c in axes_temp:
        handles,labels = c.get_legend_handles_labels() 
        ltemp = {label: handle for handle, label in zip(handles,labels)}
        labels_handles.update(ltemp)
        
      return labels_handles #labels_handles.values(),labels_handles.keys()

  def plot_joint_panel(self,ax,time,jointout,xout,refsol=None):
      ax, ax_pmarginal,ax_qmarginal = self.scatter_panel_layout(ax,time)

      self.joint_distributions_scatter_panel(ax,
                                              ax_pmarginal,
                                              ax_qmarginal,
                                              jointout,
                                              xout,time,vmax=0.5,refsol=refsol)

if __name__=="__main__":
  
  update_mpl()
