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

if __name__=="__main__":
  
  update_mpl()
