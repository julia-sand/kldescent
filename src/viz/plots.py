import os
import string

import pdb

import numpy as np 
from sklearn.neighbors import KernelDensity
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

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
  mpl.rcParams['xtick.labelsize'] = 18
  mpl.rcParams['ytick.labelsize'] = 18
  mpl.rcParams['font.size'] = 22
  mpl.rcParams['axes.labelsize'] = 22

class PlotFuns():
  def __init__(self,boundary_data=None):

    update_mpl()

    #self.subdir = self.folder #add default value to put plots into main folder if subdir has not been set
    self.boundary = boundary_data
    self.folder = os.path.dirname(__file__)
    #self.epsilon = 1 #if boundary_data==None else boundary_data.params["epsilon"]

    #borders of xaxis
    self.qmax = 6
    
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
  
  
  def setresultsdir(self,folder):
    #set output location 
    self.folder = folder
    self.subdir = folder #also set the subdirectory to default to folder name

  def setsubdir(self, subdir):
    self.subdir = self.folder+subdir #os.path.join(self.folder,subdir)

  def make_plot_label(self,plot_index):
    return "("+string.ascii_lowercase[plot_index]+")"

  #function for plotting
  def joint_distributions_scatter(self,fig,
                                  gs_joint_distributions,
                                  plot_index,
                                  joint_out,
                                  x_samples,
                                  time=None,vmax=0.5):
    """
    Function that plots the scatter plot of the monte carlo approximation of the joint distribution

    input:
    - fig: the figure for plotting
    - plot_index: location of plot
    - joint_out: joint distribution as a numpy array
    - Q: q-coordinate
    - P: p-coordinate
    - time
    - vmax: maximum value for the scatter plots
    """

    x_ind = int(np.floor(plot_index/3))
    y_ind = plot_index % 3

    plot_title_value = round(time/self.boundary.params["Tf"],2)

    #get plot location
    ax = fig.add_subplot(gs_joint_distributions[x_ind,y_ind])
    ax.text(-9,0.4,self.make_plot_label(plot_index),zorder = 200)

    #plotting goes here
    #self.joint_distributions_scatter_panel(ax,joint_out,x_samples,time,vmax)
    
    #add boundary conditions to first plot and format titles
    if plot_index ==0:
      ax.set_title(f"$t = 0$", loc = "center")
      ax_pmarginal.fill_between(q_init,np.exp(-(q_init**2)/2)/np.sqrt(np.pi*2),color=self.c2)
      ax_qmarginal.fill_between(self.boundary.q_marginal_initial(q_init),q_init,color=self.c2)
    elif plot_index == 5:
      ax.set_title(f"$t = t_f$", loc = "center")
      #ax_pmarginal.fill_between(q_init,np.exp(-self.boundary.beta*(q_init**2)/2)/np.sqrt(np.pi*2/self.boundary.beta),color=self.c2)
      #ax_qmarginal.fill_between(self.boundary.q_marginal_final(q_init),q_init,color=self.c2)
    else:
      ax.set_title(f"$t = {plot_title_value} t_f$", loc = "center")

  def plot_marginal_joint(self,ax,x0_marginal,x0_init):

    ax.plot(x0_init,x0_marginal#/xnorm
                      ,color=self.c1)

  def normalise(self,x,y):
    idx = np.argsort(x)

    return y/np.trapz(y[idx],x[idx])

  def joint_distributions_scatter_panel(self,
                                        ax,
                                        joint_out,
                                        xs,
                                        time=None,vmax=0.5):
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
      time: float
        time at which the plot is evaluated- defaults to final time
      vmax: float
        maximum value for the scatter plots color scale
    """

    #pdb.set_trace()
    
    #set default time
    if time==None:
      time = self.boundary.params["Tf"]
    
    #make inset axes to show marginal distributions
    ax_pmarginal = ax.inset_axes([0, 1.05, 1, 0.6])
    ax_qmarginal = ax.inset_axes([-0.65, 0, 0.6, 1])

    n = joint_out.shape[0]
    
    joint_out_temp = joint_out.reshape((int(np.sqrt(n)),int(np.sqrt(n))))
    Q = xs[...,[0]].reshape((int(np.sqrt(n)),int(np.sqrt(n))))
    p_init = Q[[0],:].flatten() #one axis 
    dx = np.abs(p_init[1]-p_init[0])

    #compute marginals
    qmarg = dx*np.sum(joint_out_temp,axis=0) 
    pmarg = dx*np.sum(joint_out_temp,axis=1) 

    print("jointmax",np.max(joint_out_temp))
    print("qmax",np.max(qmarg))
    print("pmax",np.max(pmarg))

    #plot and format marginals
    self.plot_marginal_joint(ax_qmarginal,p_init,self.normalise(p_init,qmarg))
    self.plot_marginal_joint(ax_pmarginal,self.normalise(p_init,pmarg),p_init)

    ax_pmarginal.set_xlim((-self.qmax,self.qmax))
    ax_pmarginal.plot(np.linspace(-self.qmax,self.qmax,),
              self.normal_distribution(np.linspace(-self.qmax,self.qmax,),mu=self.mut_p,sigmasq=self.sigsqt_p),color="orange")
    
    ax_qmarginal.set_ylim((-self.qmax,self.qmax))
    ax_qmarginal.plot(
              self.normal_distribution(np.linspace(-self.qmax,self.qmax,),mu=self.mut_q,sigmasq=self.sigsqt_q),np.linspace(-self.qmax,self.qmax,),color="orange")
    #ax_qmarginal.set_xlim((-0.05,0.7))
    ax_qmarginal.invert_xaxis()

    #plot joint distribution
    ax.scatter(xs[...,0],xs[...,1],c=joint_out_temp.T.flatten()
              ,cmap="Blues",norm="log",vmin=0.001,vmax=vmax)
    
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

    #ax.set_xlim((-self.qmax,self.qmax))
    #ax.set_ylim((-self.qmax,self.qmax))

    ax_pmarginal.set_xticklabels([])
   
    ax.set_xlabel(r"$p$")
    ax_qmarginal.set_xlabel(r"$\rho_t(q)$")

    ax.set_ylabel(r"$q$")
    ax_pmarginal.set_ylabel(r"$\rho_t(p)$")
    ax.yaxis.set_label_position("right")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-2.5,0,2.5])
    

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

  def make_legend(self,fig):
    #create the legend 
    blue_line = mlines.Line2D([], [],color=self.c1)
    #orange_line = mlines.Line2D([], [],color="orange",lw=self.lw)
    #green_line = mlines.Line2D([], [],color="green",lw=self.lw,linestyle="dashed")
    blue_patch = mpatches.Patch(color = self.c2, alpha=1)

    legend = fig.legend(handles=[blue_line,
                                  #orange_line,
                                  #green_line,
                                  blue_patch],
                        labels = ["Marginal Distributions",
                                  #"Perturbative Prediction",
                                  #"Gaussian Prediction",
                                  "Assigned Boundary Conditions"],
            frameon = False,
            handlelength = 2,
            ncols = 2,
            bbox_to_anchor=(0.75,0.2))

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.85,0.25))

  def make_legend_marginals(self,fig):
    #create the legend 
    blue_line = mlines.Line2D([], [],color=self.c1)
    orange_line = mlines.Line2D([], [],color="orange")
    green_line = mlines.Line2D([], [],color="green")
    
    legend = fig.legend(handles=[blue_line,
                                  orange_line,
                                  green_line],
                        labels = ["Reference",
                                  "Controlled by NN.",
                                  "Free Diffusion"],
            frameon = False,
            handlelength = 1,
            ncols = 3,bbox_to_anchor=(0.5, 0.1),
            loc='lower center')

    #move the legend
    #legend.set_bbox_to_anchor(bbox=(0.85,-0.25))

  def format_drift_ax(self,ax1,time):
    ax1.set_xlim((-self.qmax,self.qmax))
    #ax1.set_xlim((-2,2))
    ax1.set_title(f"t={time}")
    ax1.set_xlabel("q")
    #ax1.set_ylim((-1.8,1.8))

  def format_ticks(self,ax):
    pass

  def normal_distribution(self,x,mu=0,sigmasq=1):
    return np.exp(-((x-mu)**2)/(2*sigmasq))/np.sqrt(2*sigmasq*np.pi)
      
  def format_distribution_ax(self,ax1):
    ax1.set_xlim((-self.qmax,self.qmax))
    #ax1.set_xlabel("q")
    ax1.set_ylim((-0.01,0.65))

  def plot_reference(self,ax1,x,du):

    ax1.plot(x,du,
            label="Reference")

  #@profile
  def plot_drift(self,times,net):

    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots(1,len(times))
  
    for t in range(len(times)):  
      
      try:
        xref,duref = self.boundary.get_reference_data(times[t])
        self.plot_reference(ax[t],xref,duref)

      except (FileNotFoundError): 
        print("The reference solution file was not found. \n \
                    Please make sure that it has been created by running the scripts in valueref.py or that it has not been moved")

      ax[t].plot(np.linspace(-self.qmax,self.qmax,1000),
                self.evaluate_net_output(net,times[t]),
                label="N.N.")

      self.format_drift_ax(ax[t],times[t])
    
      ax[t].set_ylabel(r"$\mathbb{U}$")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(self.subdir+f"/drift.png", bbox_inches='tight')
    plt.close()

  def fit_kde(self,xs):

    try:

      kde = KernelDensity(kernel = "epanechnikov",bandwidth=0.1).fit(xs[np.isfinite(xs)].reshape(-1,1))
      return kde.score_samples(np.linspace(-self.qmax,self.qmax,num =1000).reshape(-1,1))

    except ValueError:
      print("The final density could not be computed because the neural network did not return any finite values.")
      pass

  ##scatter plot of final distribution 
  def marginal_distribution(self,xs,plottime):

    """
    Function that plots the scatter plot of the monte carlo approximation of the joint distribution

    input:
    - samples: the output of the evolutions
    """

    fig = plt.figure(figsize=(10,8))
    #get plot location
    ax_pos = fig.add_subplot(121)
    ax_mom = fig.add_subplot(122)

    #ax_pos.plot(np.linspace(-self.qmax,self.qmax),
    #          self.boundary.q_marginal_final(np.linspace(-self.qmax,self.qmax)))

    ax_mom.plot(np.linspace(-self.qmax,self.qmax,),
              self.normal_distribution(np.linspace(-self.qmax,self.qmax,),mu=0,sigmasq=1),)
              #np.exp(-np.linspace(-self.qmax,self.qmax,)**2)/np.sqrt(np.pi))

    ax_pos.set_title("Position")
    ax_mom.set_title("Momentum")
    colors = ["orange",self.c1]
    labels = ["Controlled by NN.","Free Diffusion"]
    for x in enumerate(xs):
    
      ax_mom.plot(np.linspace(-self.qmax,self.qmax,num=1000), np.exp(self.fit_kde(x[1][:,1])), color=colors[x[0]],label=labels[x[0]])
      ax_pos.plot(np.linspace(-self.qmax,self.qmax,num=1000), np.exp(self.fit_kde(x[1][:,0])), color=colors[x[0]],label=labels[x[0]])
      
      ax_pos.plot(np.linspace(-self.qmax,self.qmax,num=1000),
          self.normal_distribution(np.linspace(-self.qmax,self.qmax,num=1000),mu=np.nanmean(x[1][:,0]),sigmasq=np.nanvar(x[1][:,0])),
          color="gray",linestyle="dashed",alpha=0.5)
      
      ax_mom.set_xlim((-self.qmax,self.qmax))
      ax_pos.set_xlim((-self.qmax,self.qmax))
    
    self.format_distribution_ax(ax_mom)
    self.format_distribution_ax(ax_pos)

    plt.suptitle(f"t = {plottime}")

    fig.subplots_adjust(bottom=0.3, wspace=0.33)

    plt.legend()
    #self.make_legend_marginals(fig)
    plt.savefig(self.subdir+"/marginals.png",bbox_inches="tight")
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
    #plt.plot(arr[:,0],arr[:,2],label=r"K.L. Divergence")

    #set axis labels
    #ax.set_xlabel("epochs")
    #ax.set_ylabel(r"Effective Loss, $\hat\mathcal{L}$")
    
    return     

  def plot_loss_metrics(self,filename=None):

    plt.figure(figsize=(15,8))

    arr = self.get_loss_output(filename)

    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    try:
      self.plot_lhat(ax1,arr)
      self.plot_dynamic_kl(ax2,arr)
      self.plot_static_kl(ax3,arr)
      plt.tight_layout()
      plt.savefig(self.folder+"/losses.png", bbox_inches='tight')
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
                                                    ,torch.linspace(-self.qmax,self.qmax,1000,requires_grad=False))))
        
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

  def plot_free_diffusion_sol(self,ax,t_vec,cumulant):
      ax.plot(t_vec,cumulant,
                            linestyle="dashed",
                            zorder=0,
                            alpha=0.5,
                            color="gray")

  def make_cumulant_subplot(self,gs_location,
                            ylabel_text,
                            t_vec,
                            cumulants_nn_plot,
                            cumulants_ref_plot,test_samples):
    
    #x = self.boundary.get_reference_timegrid()
    #get index:
    ax = plt.subplot(gs_location)

    if ylabel_text=="Position Mean":
      index = 0 
      ax.errorbar(t_vec, cumulants_nn_plot[:,index], yerr=np.sqrt(cumulants_nn_plot[:,2]/test_samples),alpha=0.5)
      self.plot_free_diffusion_sol(ax,t_vec,self.position_mean_free_diffusion(t_vec,cumulants_ref_plot[0,index],cumulants_ref_plot[0,1]))

    elif ylabel_text=="Momentum Mean":
      index = 1
      ax.errorbar(t_vec, cumulants_nn_plot[:,index], yerr=np.sqrt(cumulants_nn_plot[:,3]/test_samples),alpha=0.5)
      self.plot_free_diffusion_sol(ax,t_vec,self.momentum_mean_free_diffusion(t_vec,cumulants_nn_plot[0,index]))

    elif ylabel_text=="Position Variance":
      index = 2 
      self.plot_free_diffusion_sol(ax,t_vec,self.position_variance_free_diffusion(t_vec,cumulants_ref_plot[0,index],cumulants_ref_plot[0,index+1]))
      ax.errorbar(t_vec, cumulants_nn_plot[:,index], yerr=cumulants_nn_plot[:,index]*np.sqrt(2/(test_samples-1)),alpha=0.5)
    elif ylabel_text=="Momentum Variance":
      index = 3 
      self.plot_free_diffusion_sol(ax,t_vec,self.momentum_variance_free_diffusion(t_vec,cumulants_ref_plot[0,index]))
      ax.errorbar(t_vec, cumulants_nn_plot[:,index], yerr=cumulants_nn_plot[:,index]*np.sqrt(2/(test_samples-1)),alpha=0.5)
    elif ylabel_text=="Cross Correlation":
      index = 4 
      self.plot_free_diffusion_sol(ax,t_vec,self.cross_correlation_free_diffusion(t_vec,cumulants_ref_plot[0,2],cumulants_ref_plot[0,3]))
      ax.errorbar(t_vec, cumulants_nn_plot[:,index], yerr=np.sqrt((1-cumulants_nn_plot[:,index]**2)/(test_samples-2)),alpha=0.5)

    ax.plot(t_vec, cumulants_nn_plot[:,index], color="orange")
    ax.plot(t_vec, cumulants_ref_plot[:,index], color=self.c1)

    try:
      t,cumulant_vals = self.boundary.get_known_solution_cumulants(ylabel_text)
      ax.plot(t,cumulant_vals,color="green")
    except (FileExistsError,FileNotFoundError): #if the known solution is somehow unavailable
      pass

    ax.set_title(self.make_plot_label(index),pad = self.titlepad,x = self.titlex, y = self.titley,zorder = 1000000)
    ax.set_ylabel(ylabel_text)
    ax.set_xlim((-0.1,t_vec[-1]+0.1))

  def plot_cumulants(self,t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples):

    # Plotting the cumulants
    fig1 = plt.figure(figsize=(15, 8))#, constrained_layout=True)

    # Create a 1x5 grid with different widths for the bottom row
    gs_cumulants = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1,1,1], height_ratios=[1, 1])
            
    # position mean
    self.make_cumulant_subplot(gs_cumulants[0, 0:2],"Position Mean",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
    self.make_cumulant_subplot(gs_cumulants[0, 2:4],"Position Variance",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
    self.make_cumulant_subplot(gs_cumulants[0, 4:],"Momentum Mean",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
    self.make_cumulant_subplot(gs_cumulants[1,:3],"Momentum Variance",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
    self.make_cumulant_subplot(gs_cumulants[1, 3:],"Cross Correlation",t_vec,cumulants_nn_plot,cumulants_ref_plot,test_samples)
                      
    plt.tight_layout()

    #get legend
    orange_line = mlines.Line2D([], [],color="orange")
    blue_line = mlines.Line2D([], [],color=self.c1)
    green_line = mlines.Line2D([], [],color="green")

    legend = fig1.legend(handles=[blue_line,orange_line,green_line],#,green_line],
              labels = ["Free Diffusion","Controlled","Known Reference Solution"],
              frameon = False,
              handlelength = 1,
              ncols = 3)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))
    plt.savefig(self.subdir+f"/cumulants.png", bbox_inches='tight')
    plt.close()

  def update_final_cumulants(self,cumulants_nn):
    self.mut_p = cumulants_nn[1]
    self.sigsqt_p = cumulants_nn[3]
    self.mut_q = cumulants_nn[0]
    self.sigsqt_q = cumulants_nn[2]
     
  def movingaverage(self, data, window_size):
      data_means = np.convolve(data, np.ones(int(window_size))/float(window_size), 'same')
      return data_means[window_size:-window_size]

  
if __name__=="__main__":
  
  update_mpl()