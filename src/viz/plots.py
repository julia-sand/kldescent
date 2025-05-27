import os

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import torch
#from memory_profiler import profile

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering

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

  ##scatter plot of final distribution 
  def joint_distribution_scatter(self,samples):

    """
    Function that plots the scatter plot of the monte carlo approximation of the joint distribution

    input:
    - samples: the output of the evolutions
    """

    fig = plt.figure()
    #get plot location
    ax = fig.add_subplot(111)

    ax.set_xlim((-4,4))
    ax.set_ylim((-4,4))

    #get the samples into numpy for plotting
    #since this was output of the neural net, we need detach. GPU will be (maybe) implemented in the future.
    #samples = samples.detach().cpu().numpy()

    #plot scatter graphs
    ax.scatter(samples[:,0],samples[:,1])
    
    Q,P = np.meshgrid(np.linspace(-5,5),np.linspace(-5,5))
    ax.contour(Q,P,
              self.boundary.p_final(Q,P))

    ax.tick_params(axis='both', labelsize=self.fontsizeticks)

    ax.set_xlabel(r"$p$",fontsize = self.fontsizetitles,labelpad=-20)
    ax.set_ylabel(r"$q$",fontsize = self.fontsizetitles,labelpad=-10)

    plt.savefig(self.subdir+f"/test_plot.png")
    plt.close()

  def format_drift_ax(self,ax1,time):
    ax1.set_xlim((-4,4))
    ax1.set_title(f"t={time}")
    ax1.set_xlabel("q")
    ax1.set_ylim((-10,10))
      
  def plot_reference(self,ax1,x,du):

    ax1.plot(x,du,
            label="Reference")

  #@profile
  def plot_drift(self,x,times,net):

    

    fig,ax = plt.subplots(1,len(times))
  
    for t in range(len(times)):  
      
      try:
        xref,duref = self.boundary.get_reference_data(times[t])
        self.plot_reference(ax[t],xref,duref)

      except FileNotFoundError: 
        print("The reference solution file was not found. \n \
                    Please make sure that it has been created by running the scripts in valueref.py or that it has not been moved")

      ax[t].plot(x,self.evaluate_net_output(net,times[t]),label="N.N.")

      self.format_drift_ax(ax[t],times[t])
    
    ax[0].set_ylabel(r"$\mathbb{U}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(self.subdir+f"/drift.png")
    plt.close()


  ##scatter plot of final distribution 
  #@profile
  def marginal_distribution(self,samples,name):

    """
    Function that plots the scatter plot of the monte carlo approximation of the joint distribution

    input:
    - samples: the output of the evolutions
    """

    try:
      kde = KernelDensity(kernel = "epanechnikov",bandwidth=0.1).fit(samples[np.isfinite(samples)].reshape(-1,1))
      log_dens = kde.score_samples(np.linspace(-6,6).reshape(-1,1))
      fig = plt.figure()
      #get plot location
      ax = fig.add_subplot(111)

      #plot scatter graphs
      ax.fill_between(np.linspace(-6,6,num=len(log_dens)), np.zeros(len(log_dens)), np.exp(log_dens), fc="#AAAAFF")
      
      ax.plot(np.linspace(-6,6,),
                np.exp(-np.linspace(-6,6,)**2)/np.sqrt(np.pi))
      
      plt.savefig(self.subdir+f"/final_density_{name}.png")
      plt.close()    

    except ValueError:
      print("The final density could not be computed because the neural network did not return any finite values.")

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
      plt.savefig(self.folder+"/losses.png")
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
                            xout,
                            net,
                            plot_times):
      #create epoch folder and set plotter subdirectory
      #make plots
      self.marginal_distribution(xout[:,0],"POSITION")
      if xout.shape[1] == 2:
        self.marginal_distribution(xout[:,1],"MOMENTUM")
      
      self.plot_drift(np.linspace(-4,4,1000),
                          times = plot_times,
                          net=net)


if __name__=="__main__":

  from ..sde.boundary import UnderdampedNoMomentum
  boundary_cond = UnderdampedNoMomentum()
  filename="tests/20250523-171743/loss.csv" #add filename of loss.csv to plot
  plotter = PlotFuns(boundary_cond)
  plotter.plot_loss_metrics(filename)
  
