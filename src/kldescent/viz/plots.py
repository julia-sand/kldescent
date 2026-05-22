import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    
def make_gridspec_subplots(fig, gs, ncols, nrows,squeeze=True):
  """
  Create a Matplotlib figure with a GridSpec-based subplot layout.

  This function mimics `plt.subplots()` but uses GridSpec internally

  Parameters
  ----------
  nrows : int
      Number of subplot rows.
  ncols : int
      Number of subplot columns.
  squeeze : bool, default=True
      If True, the returned axes are squeezed to match `plt.subplots()` behavior:
      - (1, 1) -> single Axes object
      - (1, N) -> 1D list of Axes
      - (N, 1) -> 1D list of Axes
      - (N, N) -> 2D list of Axes
      If False, always returns a 2D list of Axes.

  Returns
  -------
  axs : Axes or list of Axes
      Subplot axes arranged according to `squeeze` behavior.
  """
  
  axs = [fig.add_subplot(gs[i, j])
        for i in range(nrows)
        for j in range(ncols)]

  axs = axs[:ncols*nrows]

  if squeeze:
      # mimic plt.subplots squeeze behavior
      if nrows == 1 and ncols == 1:
          return axs[0][0]
      elif nrows == 1:
          return axs[0]
      elif ncols == 1:
          return [axs[i][0] for i in range(nrows)]
      else:
          return axs

  return axs

def make_fig_subplots(nrows,ncols,squeeze=True):

  fig,axs = plt.subplots(ncols=ncols,nrows=nrows,squeeze=squeeze)
  fig.set_figheight(12)
  fig.set_figwidth(25)

  return fig,axs

def make_legend(fig,handles,labels): 

  fig.legend(handles,labels,
                      frameon = False,
                      handlelength = 1,
                      ncols = min(4,len(labels)),
                      loc="lower center")

def save_fig_(fig,filename,loc):
  fig.savefig(loc+"/"+filename+".png", bbox_inches='tight')
  fig.savefig(loc+"/"+filename+".eps",format="eps", bbox_inches='tight')
  fig.savefig(loc+"/"+filename+".pdf",format="pdf", bbox_inches='tight')
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
  mpl.rcParams['xtick.labelsize'] = 22
  mpl.rcParams['ytick.labelsize'] = 22
  mpl.rcParams['font.size'] = 24
  mpl.rcParams['axes.labelsize'] = 24

if __name__=="__main__":
  
  update_mpl()
