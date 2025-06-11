import os
import string

import numpy as np 
from sklearn.neighbors import KernelDensity
import torch

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

'''
#visualisations ideas

1. Plot of the drift compared to the reference at the start of training
2. Plot of the KL divergence of final boundary conditions vs training epoch
3. Plot histogram/scatter of final probability distribution 
4. NN drift vs perturbative during training. 
5. loss vs epoch
'''

class PlotFuns():
  def __init__(self,boundary_data=None):

    #self.subdir = self.folder #add default value to put plots into main folder if subdir has not been set
    self.boundary = boundary_data
    self.folder = os.path.dirname(__file__)
    
    #borders of xaxis
    self.qmax = 6
    
    #fontsizes
    self.fontsize = 22
    self.fontsizeticks = 18
    self.fontsizetitles = 22

    #linewidth
    self.lw = 3

    #colors
    self.c1 = "#1f78b4" #darkblue
    self.c2 = "#a6cee3" #lightblue
    self.c3 = "#33a02c" #dark green
    self.c4 = "#b2df8a" #light green

    self.titlepad = 5
    self.titlex = 0.07
    self.titley = 0.88

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
                                  time,vmax=0.5):

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

    pmax,qmax = 4,4
    p_init = np.linspace(-pmax,pmax,x_samples)
    q_init = np.linspace(-qmax,qmax,x_samples)
    
    Q,P = np.meshgrid(q_init,p_init)

    x_ind = int(np.floor(plot_index/3))
    y_ind = plot_index % 3

    plot_title_value = round(time/self.boundary.Tf,2)

    #get plot location
    ax = fig.add_subplot(gs_joint_distributions[x_ind,y_ind])

    ax_pmarginal = ax.inset_axes([0, 1.05, 1, 0.6])
    ax_qmarginal = ax.inset_axes([-0.65, 0, 0.6, 1])
    ax_pmarginal.text(-9,0.4,self.make_plot_label(plot_index),fontsize = self.fontsizetitles,zorder = 200)

    #compute joint distribution
    idx = np.argwhere(~np.isnan(joint_out))
    
    pmarginal = np.nansum(joint_out.reshape((x_samples,x_samples)),axis=1)
    porder = np.argsort(p_init)
    pnorm = np.trapz(pmarginal[porder],p_init[porder])
    ax_pmarginal.plot(p_init[porder],pmarginal[porder]/pnorm
                      ,color=self.c1,lw=self.lw)


    qmarginal = np.nansum(joint_out.reshape((x_samples,x_samples)),axis=0)#[np.nansum(joint_out[i::samples]) for i in range(0,samples)]
    qorder = np.argsort(q_init)
    qnorm = np.trapz(qmarginal[qorder],q_init[qorder])

    ax_qmarginal.plot(qmarginal[qorder]/qnorm
                      ,q_init[qorder],color=self.c1,lw=self.lw)

    ax_pmarginal.set_xlim((-pmax,pmax))
    ax_pmarginal.set_ylim((-0.05,0.7))

    ax_qmarginal.set_ylim((-qmax,qmax))
    ax_qmarginal.set_xlim((-0.05,0.7))

    #plot scatter graphs
    zout = joint_out[idx]#/max(joint_out[idx])

    order = np.argsort(zout.flatten())
    pout = P.flatten()[idx].flatten()
    qout = Q.flatten()[idx].flatten()
    ax.scatter(pout[order],qout[order],c=zout.flatten()[order]
              ,cmap="Blues",norm="log",vmin=0.001,vmax=vmax)

    ###ticks formatting and labelling
    ax_pmarginal.yaxis.tick_right()
    ax.yaxis.tick_right()
    
    ax_pmarginal.tick_params(
        axis='y',
        which='both',
        length=5,
        rotation=-20,
        pad = -3,
        labelsize = self.fontsizeticks)
    ax_pmarginal.tick_params(
        axis='x',
        which='both',
        labelsize = self.fontsizeticks)
    ax_qmarginal.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',
        rotation=70,
        labelsize = self.fontsizeticks,
        length =5,
        pad = -3,
        labelleft=False)
    ax_qmarginal.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      #both major and minor ticks
        left=False,
        labelleft=False)

    ax.set_xlim((-pmax,pmax))
    ax.set_ylim((-qmax,qmax))

    ax.tick_params(axis='both', 
                    labelsize=self.fontsizeticks)
    ax_pmarginal.set_xticklabels([])
   
    ax.set_xlabel(r"$p$",fontsize = self.fontsizetitles)
    ax_qmarginal.set_xlabel(r"$\rho_t(q)$",fontsize = self.fontsizetitles)

    ax.set_ylabel(r"$q$",fontsize = self.fontsizetitles)
    ax_pmarginal.set_ylabel(r"$\rho_t(p)$",fontsize = self.fontsizetitles)
    ax.yaxis.set_label_position("right")
    #add boundary conditions to first plot and format titles
    if plot_index ==0:
      ax.set_title(f"$t = 0$", loc = "center", fontsize=self.fontsizetitles)
      ax_pmarginal.fill_between(q_init,np.exp(-self.boundary.beta*(q_init**2)/2)/np.sqrt(np.pi*2/self.boundary.beta),color=self.c2)
      ax_qmarginal.fill_between(self.boundary.q_marginal_initial(q_init),q_init,color=self.c2)
    elif plot_index == 5:
      ax.set_title(f"$t = t_f$", loc = "center", fontsize=self.fontsizetitles)
      #ax_pmarginal.fill_between(q_init,np.exp(-self.boundary.beta*(q_init**2)/2)/np.sqrt(np.pi*2/self.boundary.beta),color=self.c2)
      #ax_qmarginal.fill_between(self.boundary.q_marginal_final(q_init),q_init,color=self.c2)
    else:
      ax.set_title(f"$t = {plot_title_value} t_f$", loc = "center", fontsize=self.fontsizetitles)

    plt.setp(ax_pmarginal.yaxis.get_majorticklabels(), va="bottom")
    plt.setp(ax_qmarginal.xaxis.get_majorticklabels(), ha="right")
    ax.set_aspect('equal', adjustable='box')
    ax_qmarginal.invert_xaxis()
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

    cb.ax.tick_params(labelsize=self.fontsizeticks)
    cb.ax.set_xlabel("Joint Distribution",fontsize=self.fontsizetitles)

  def make_legend(self,fig):
    #create the legend 
    blue_line = mlines.Line2D([], [],color=self.c1,lw=self.lw)
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
            fontsize = self.fontsizetitles,
            frameon = False,
            handlelength = 2,
            ncols = 2,
            bbox_to_anchor=(0.75,0.2))

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.85,0.25))

  def make_legend_marginals(self,fig):
    #create the legend 
    blue_line = mlines.Line2D([], [],color=self.c1,lw=self.lw)
    orange_line = mlines.Line2D([], [],color="orange",lw=self.lw)
    green_line = mlines.Line2D([], [],color="green",lw=self.lw)
    
    legend = fig.legend(handles=[blue_line,
                                  orange_line,
                                  green_line],
                        labels = ["Reference",
                                  "Controlled by NN.",
                                  "Free Diffusion"],
            fontsize = self.fontsizeticks,
            frameon = False,
            handlelength = 1,
            ncols = 3,bbox_to_anchor=(0.5, 0.1),
            loc='lower center')

    #move the legend
    #legend.set_bbox_to_anchor(bbox=(0.85,-0.25))

  def format_drift_ax(self,ax1,time):
    ax1.set_xlim((-self.qmax,self.qmax))
    ax1.set_title(f"t={time}")
    ax1.set_xlabel("q")
    ax1.set_ylim((-1.8,1.8))

  def format_ticks(self,ax):
    ax.tick_params(which="both",axis="both",labelsize=self.fontsizeticks)

  def normal_distribution(self,x,mu=0,sigmasq=1):
    return np.exp(-((x-mu)**2)/(2*sigmasq))/np.sqrt(2*sigmasq*np.pi)
      
  def format_distribution_ax(self,ax1):
    ax1.set_xlim((-self.qmax,self.qmax))
    #ax1.set_xlabel("q")
    ax1.set_ylim((-0.01,0.65))
    ax1.tick_params(which="both",axis="both",labelsize=self.fontsizeticks)

  def plot_reference(self,ax1,x,du):

    ax1.plot(x,du,
            label="Reference")

  #@profile
  def plot_drift(self,times,net):

    fig,ax = plt.subplots(1,len(times))
  
    for t in range(len(times)):  
      
      try:
        xref,duref = self.boundary.get_reference_data(times[t])
        self.plot_reference(ax[t],xref,duref)

      except FileNotFoundError: 
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
              self.normal_distribution(np.linspace(-self.qmax,self.qmax,),mu=0,sigmasq=1/self.boundary.beta),)
              #np.exp(-np.linspace(-self.qmax,self.qmax,)**2)/np.sqrt(np.pi))

    ax_pos.set_title("Position",fontsize=self.fontsizetitles)
    ax_mom.set_title("Momentum",fontsize=self.fontsizetitles)
    colors = ["orange",self.c1]
    labels = ["Controlled by NN.","Free Diffusion"]
    for x in enumerate(xs):
    
      ax_mom.plot(np.linspace(-self.qmax,self.qmax,num=1000), np.exp(self.fit_kde(x[1][:,1])), color=colors[x[0]],label=labels[x[0]])
      ax_pos.plot(np.linspace(-self.qmax,self.qmax,num=1000), np.exp(self.fit_kde(x[1][:,0])), color=colors[x[0]],label=labels[x[0]])
      
      ax_pos.plot(np.linspace(-self.qmax,self.qmax,num=1000),
          self.normal_distribution(np.linspace(-self.qmax,self.qmax,num=1000),mu=np.nanmean(x[1][:,0]),sigmasq=np.nanvar(x[1][:,0])),
          color="gray",lw=self.lw,linestyle="dashed",alpha=0.5)
      
      ax_mom.set_xlim((-self.qmax,self.qmax))
      ax_pos.set_xlim((-self.qmax,self.qmax))
    
    self.format_distribution_ax(ax_mom)
    self.format_distribution_ax(ax_pos)

    plt.suptitle(f"t = {plottime}",fontsize=self.fontsizetitles)

    fig.subplots_adjust(bottom=0.3, wspace=0.33)

    plt.legend(fontsize=self.fontsizetitles)
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

  def plot_lhat(self,ax,arr):


    ax.plot(arr[:,0],arr[:,1],label=r"$\hat\mathcal{L}$")
    #plt.plot(arr[:,0],arr[:,2],label=r"K.L. Divergence")

    #set axis labels
    ax.set_xlabel("epochs")
    ax.set_ylabel(r"Effective Loss, $\hat\mathcal{L}$")
    
    return     

  def plot_dynamic_kl(self,ax,arr):

    #arr = self.get_loss_output()

    ax.plot(arr[:,0],arr[:,2],label=r"K.L. Divergence")

    #set axis labels
    ax.set_xlabel("epochs")
    ax.set_ylabel(r"(dynamic) K.L.")

  def plot_static_kl(self,ax,arr):

    #arr = self.get_loss_output()

    ax.plot(arr[:,0],arr[:,3],label=r"K.L. Divergence")

    #set axis labels
    ax.set_xlabel("epochs")
    ax.set_ylabel(r"(static) K.L.")

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
      for ax in [ax1,ax2,ax3]:
        ax.set_ylim((-2.5,2.5))
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
      #computes neural network output (forward pass) to use in plotting
      return net(torch.column_stack((time*torch.ones(1000,dtype=torch.float32)
                                                  ,torch.linspace(-self.qmax,self.qmax,1000)))).detach().numpy().reshape((1000,1))
      



  def make_cumulant_subplot(self,gs_location,
                            ylabel_text,t_vec,cumulants_nn_plot,cumulants_ref_plot):
    
    #get index:
    if ylabel_text=="Position Mean":
      index = 0 
    elif ylabel_text=="Momentum Mean":
      index = 1
    elif ylabel_text=="Position Variance":
      index = 2 
    elif ylabel_text=="Momentum Variance":
      index = 3 
    elif ylabel_text=="Cross Correlation":
      index = 4 

    # position mean
    plt.subplot(gs_location)
    plt.plot(t_vec, cumulants_nn_plot[:,index], lw=self.lw,color="orange")
    plt.plot(t_vec, cumulants_ref_plot[:,index], lw=self.lw,color=self.c1)
    try:
      plt.plot(self.boundary.get_known_solution_cumulants(ylabel_text)[0],
                self.boundary.get_known_solution_cumulants(ylabel_text)[1],lw=self.lw,color="green")
    except (ValueError,TypeError):
      pass
    plt.title(self.make_plot_label(index),fontsize = self.fontsizetitles,pad = self.titlepad,x = self.titlex, y = self.titley,zorder = 1000000)
    plt.ylabel(ylabel_text,fontsize = self.fontsizetitles)
    plt.xlim((-0.1,t_vec[-1]+0.1))
    self.format_ticks(plt.gca())    

  def plot_cumulants(self,t_vec,cumulants_nn_plot,cumulants_ref_plot):

    # Plotting the cumulants
    fig1 = plt.figure(figsize=(15, 8))#, constrained_layout=True)

    # Create a 1x5 grid with different widths for the bottom row
    gs_cumulants = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1,1,1], height_ratios=[1, 1])
            
    # position mean
    self.make_cumulant_subplot(gs_cumulants[0, 0:2],"Position Mean",t_vec,cumulants_nn_plot,cumulants_ref_plot)
    self.make_cumulant_subplot(gs_cumulants[0, 2:4],"Position Variance",t_vec,cumulants_nn_plot,cumulants_ref_plot)
    self.make_cumulant_subplot(gs_cumulants[0, 4:],"Momentum Mean",t_vec,cumulants_nn_plot,cumulants_ref_plot)
    self.make_cumulant_subplot(gs_cumulants[1,:3],"Momentum Variance",t_vec,cumulants_nn_plot,cumulants_ref_plot)
    self.make_cumulant_subplot(gs_cumulants[1, 3:],"Cross Correlation",t_vec,cumulants_nn_plot,cumulants_ref_plot)
                      
    plt.tight_layout()

    #get legend
    orange_line = mlines.Line2D([], [],color="orange",lw=self.lw)
    blue_line = mlines.Line2D([], [],color=self.c1,lw=self.lw)
    green_line = mlines.Line2D([], [],color="green",lw=self.lw)

    legend = fig1.legend(handles=[blue_line,orange_line,green_line],#,green_line],
              labels = ["Free Diffusion","Controlled","Known Reference Solution"],
              fontsize = self.fontsizetitles,
              frameon = False,
              handlelength = 1,
              ncols = 3)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))
    plt.savefig(self.subdir+f"/cumulants.png", bbox_inches='tight')
    plt.close()


if __name__=="__main__":

  from ..sde.boundary import *
  boundary_cond = UnderdampedHalfGaussian()
  fileloc=f"tests/{boundary_cond.__class__.__name__}"
  filename = fileloc+"/loss.csv" #add filename of loss.csv to plot
  plotter = PlotFuns(boundary_cond)
  plotter.setresultsdir(fileloc)
  plotter.plot_loss_metrics(filename)
  
