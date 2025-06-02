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

  def setresultsdir(self,folder):
    #set output location 
    self.folder = folder
    self.subdir = folder #also set the subdirectory to default to folder name

  def setsubdir(self, subdir):
    self.subdir = self.folder+subdir #os.path.join(self.folder,subdir)

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
    ax_pmarginal.text(-9,0.4,"("+string.ascii_lowercase[plot_index]+")",fontsize = self.fontsizetitles,zorder = 200)

    #compute joint distribution
    idx = np.argwhere(~np.isnan(joint_out))
    
    pmarginal = np.nansum(joint_out.reshape((x_samples,x_samples)),axis=1)
    porder = np.argsort(p_init)
    pnorm = 1#np.trapz(pmarginal[porder],p_init[porder])
    ax_pmarginal.plot(p_init[porder],pmarginal[porder]/pnorm
                      ,color=self.c1,lw=self.lw)


    qmarginal = np.nansum(joint_out.reshape((x_samples,x_samples)),axis=0)#[np.nansum(joint_out[i::samples]) for i in range(0,samples)]
    qorder = np.argsort(q_init)
    qnorm = 1#np.trapz(qmarginal[qorder],q_init[qorder])

    ax_qmarginal.plot(qmarginal[qorder]/qnorm
                      ,q_init[qorder],color=self.c1,lw=self.lw)

    #self.format_dist_axes(ax_pmarginal)

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

  def format_drift_ax(self,ax1,time):
    ax1.set_xlim((-2,2))
    ax1.set_title(f"t={time}")
    ax1.set_xlabel("q")
    ax1.set_ylim((-10,10))
      
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

      ax[t].plot(np.linspace(-4,4,1000),
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
      return kde.score_samples(np.linspace(-6,6,num =1000).reshape(-1,1))

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

    fig = plt.figure()
    #get plot location
    ax_pos = fig.add_subplot(121)
    ax_mom = fig.add_subplot(122)

    ax_pos.plot(np.linspace(-6,6,),
              self.boundary.q_marginal_final(np.linspace(-6,6,)))

    ax_mom.plot(np.linspace(-6,6,),
              np.exp(-np.linspace(-6,6,)**2)/np.sqrt(np.pi))

    ax_pos.set_title("Position")
    ax_mom.set_title("Momentum")

    for x in xs:

      #plot scatter graphs
      ax_mom.plot(np.linspace(-6,6,num=1000), np.exp(self.fit_kde(x[:,1])))
      ax_pos.plot(np.linspace(-6,6,num=1000), np.exp(self.fit_kde(x[:,0])))
    
    ax_mom.set_xlim((-4,4))
    ax_pos.set_xlim((-4,4))
    plt.suptitle(f"t = {plottime}")
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

    ax.plot(arr[:,0],(self.boundary.beta/4)*arr[:,2],label=r"K.L. Divergence")

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
                                                  ,torch.linspace(-4,4,1000)))).detach().numpy().reshape((1000,1))
      

  #@profile
  def make_epoch_plots(self,
                            xs,
                            net,
                            plot_times):
      #create epoch folder and set plotter subdirectory
      #make plots
      self.marginal_distribution(xs,plottime)
      
      self.plot_drift(times = plot_times,
                          net=net)


if __name__=="__main__":

  from ..sde.boundary import UnderdampedNoMomentum
  boundary_cond = UnderdampedNoMomentum()
  filename="tests/20250523-171743/loss.csv" #add filename of loss.csv to plot
  plotter = PlotFuns(boundary_cond)
  plotter.plot_loss_metrics(filename)
  
