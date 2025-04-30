import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os 
from scipy.ndimage import generic_filter,median

'''
#visualisations ideas

1. Plot of the drift compared to the reference at the start of training
2. Plot of the KL divergence of final boundary conditions vs training epoch
3. Plot histogram/scatter of final probability distribution 
4. NN drift vs perturbative during training. 
5. loss vs epoch
'''

class PlotFuns():
  def __init__(self,boundary_data):

    #self.subdir = self.folder #add default value to put plots into main folder if subdir has not been set
    self.boundary = boundary_data

    #get reference solution location 
    self.dirname = os.path.dirname(__file__)

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

  def format_drift_ax(self,ax1,time):
    ax1.set_xlim((-4,4))
    ax1.set_title(f"t={time}")
    ax1.set_xlabel("q")
    #ax1.set_ylim((-10,10))
      
  def plot_reference(self,ax1,x,du):

    ax1.plot(x,du,
            label="Reference")


  def plot_drift(self,x,times,dus=None):

    fig,ax = plt.subplots(1,len(times))
  
    for t in range(len(times)):  
          
      xref,duref = self.boundary.get_reference_data(times[t])
      self.plot_reference(ax[t],xref,duref)

      for du in dus[t]:

        ax[t].plot(x,du,label="N.N.")

      self.format_drift_ax(ax[t],times[t])
    
    ax[0].set_ylabel(r"$\mathbb{U}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(self.subdir+f"/drift.png")
    plt.close()

  #plot the drift used in pretraining
  def plot_pretrainsol(self,times):
    
    fig,ax = plt.subplots(1,len(times))
    for t in range(len(times)):      
      arr = np.loadtxt("../../results.csv",
                    delimiter=",", skiprows=1)
    
      arr = arr[np.where(np.abs(arr[:,1])<4)]
      ##fetch the perturbative underdamped approximation data
    
      arr = arr[np.where(arr[:,0]==np.round(times[t],5))]
      arr = arr[arr[:,1].argsort()] #sort by x

      #plot reference
      ax[t].plot(arr[:,1],arr[:,-1],label="Reference")
      #label="Reference")

      ax[t].set_title(f"t={times[t]}")
      ax[t].set_xlabel("q")
      #ax[t].set_ylim((-10,10))
    
    ax[0].set_ylabel(r"$\mathbb{U}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(self.subdir+f"/drift.png")
    plt.close()



  ##scatter plot of final distribution 
  def marginal_distribution(self,samples):

    """
    Function that plots the scatter plot of the monte carlo approximation of the joint distribution

    input:
    - samples: the output of the evolutions
    """

    fig =plt.figure()
    #get plot location
    ax = fig.add_subplot(111)

    ax.set_xlim((-4,4))
    ax.set_ylim(bottom = -0.01)

    #plot scatter graphs
    #ax.hist(samples,density =True)
    
    #fit kde 
    # tophat KDE
    from sklearn.neighbors import KernelDensity
    try:

      kde = KernelDensity(kernel = "epanechnikov",bandwidth=0.1).fit(samples[~np.isnan(samples)].reshape(-1,1))
      log_dens = kde.score_samples(np.linspace(-4,4).reshape(-1,1))
      ax.fill(np.linspace(-4,4), np.zeros(len(log_dens)), np.exp(log_dens), fc="#AAAAFF")
    except:
      print("Current gradient returns only NAN values.")
      pass

    #Q,P = np.meshgrid(np.linspace(-5,5),np.linspace(-5,5))
    ax.plot(np.linspace(-4,4),
              self.boundary.q_marginal_final(np.linspace(-4,4)),
              label="Assigned Final Distribution")
    ax.plot(np.linspace(-4,4),
              (1/np.sqrt(2*np.pi))*np.exp(-(1/2)*np.linspace(-4,4)**2),
              label="Normal(0,1)")

    ax.tick_params(axis='both', labelsize=self.fontsizeticks)

    #if x_ind !=0:
    ax.set_xlabel(r"$p$",fontsize = self.fontsizetitles,labelpad=-20)

    #if y_ind == 2:
    #make label and add text
    ax.set_ylabel(r"$\rho_T(q)$",fontsize = self.fontsizetitles,labelpad=-10)

    plt.legend()
    plt.savefig(self.subdir+f"/final_density.png")
    plt.close()

if __name__=="__main__":
  #os.makedirs("../../tests"+"/pretrain") #add epoch plots folder
  
  plotter = PlotFuns("../../tests")
  plotter.setsubdir("pretrain")
  
  plotter.plot_pretrainsol(times=[0,0.1,0.2])
