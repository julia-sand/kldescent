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
  def __init__(self,folder):

    #set output location 
    self.folder = folder

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

  ##scatter plot of final distribution 
  def joint_distribution_scatter(self,samples,epoch=1):

    """
    Function that plots the scatter plot of the monte carlo approximation of the joint distribution

    input:
    - samples: the output of the evolutions
    """

    fig =plt.figure()
    #get plot location
    ax = fig.add_subplot(111)

    ax.set_xlim((-5,5))
    ax.set_ylim((-5,5))

    #get the samples into numpy for plotting
    #since this was output of the neural net, we need detach. GPU will be (maybe) implemented in the future.
    #samples = samples.detach().cpu().numpy()

    #plot scatter graphs
    ax.scatter(samples[:,0],samples[:,1])
    
    import utils.boundary as boundary
    Q,P = np.meshgrid(np.linspace(-5,5),np.linspace(-5,5))
    ax.contour(Q,P,
              boundary.p_final(Q,P))

    ax.tick_params(axis='both', labelsize=self.fontsizeticks)

    #if x_ind !=0:
    ax.set_xlabel(r"$p$",fontsize = self.fontsizetitles,labelpad=-20)

    #if y_ind == 2:
    #make label and add text
    ax.set_ylabel(r"$q$",fontsize = self.fontsizetitles,labelpad=-10)

    plt.savefig(self.folder+f"/epoch{epoch}/test_plot.png")

  #plot the drift at time t=0  
  #TODO: update to show multiple panels with different times/formatting
  def plot_drift(self,x,dus,times,epoch=0):
    

    fig,ax = plt.subplots(1,len(times))
    for t in range(len(times)):      
      
      ##fetch the perturbative underdamped approximation data
      arr = np.loadtxt(os.path.join(self.dirname, "../refsolution/value.csv"),
                      delimiter=",")  ##TODO find a better way to do this without reloading the file eachtime. 
      #select all zeros
      arr = arr[np.where(arr[:,0]==np.round(times[t],5))]
      arr = arr[arr[:,1].argsort()] #sort by x

      #plot reference
      ax[t].plot(arr[:,1],generic_filter(arr[:,-1],median,50,mode="nearest"),label="Reference")

      for du in dus[t]:

        ax[t].plot(x,du,label="N.N.")

      ax[t].set_title(f"t={times[t]}")
      ax[t].set_xlabel("q")
    
    ax[0].set_ylabel("$\mathbb{U}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(self.folder+f"/epoch{epoch}/drift.png")
    plt.close()

  ##scatter plot of final distribution 
  def marginal_distribution(self,samples,epoch=1):

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
    #ax.hist(samples)
    
    #fit kde 
    # tophat KDE
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel = "epanechnikov",bandwidth=0.1).fit(samples.reshape(-1,1))
    log_dens = kde.score_samples(np.linspace(-4,4).reshape(-1,1))
    ax.fill(np.linspace(-4,4), np.exp(log_dens), fc="#AAAAFF")


    import utils.boundary as boundary
    #Q,P = np.meshgrid(np.linspace(-5,5),np.linspace(-5,5))
    ax.plot(np.linspace(-4,4),
              boundary.q_marginal_final(np.linspace(-4,4)))

    ax.tick_params(axis='both', labelsize=self.fontsizeticks)

    #if x_ind !=0:
    ax.set_xlabel(r"$p$",fontsize = self.fontsizetitles,labelpad=-20)

    #if y_ind == 2:
    #make label and add text
    ax.set_ylabel(r"$\rho_T(q)$",fontsize = self.fontsizetitles,labelpad=-10)

    plt.savefig(self.folder+f"/epoch{epoch}/final_density.png")
    plt.close()

if __name__=="__main__":
  
  #from boundary import *
  #from sampler import generate_samples
  #xout = generate_samples(100,mode="initial")
  #PlotFuns().joint_distribution_scatter(xout.detach().cpu().numpy(),99)
  pass