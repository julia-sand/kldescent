import os


import numpy as np 

import matplotlib as mpl
import matplotlib.pyplot as plt

from kldescent.viz.plots import update_mpl
from kldescent.utils.parser import fetch_results_folder_from_cmd


def movingaverage(data, window_size):
  """compute moving average along array
  args:
    data : 1darray
      one-dimensional array
    window_size : int
      size of filter for moving average
  returns: 
    data : 1darray
      smoothed data same shape as input
  """

  data_means = np.convolve(data, np.ones(int(window_size))/float(window_size), 'same')
  return data_means[window_size:-window_size]

def get_loss_output(folder,filename=None):
  """Open csv with results"""
  arr = np.loadtxt(folder+"/loss.csv",
                      delimiter=",",skiprows=1)  

  return arr

def plot_metric(ax,xdata,ydata,ylabel,window_size=5):
  """Plot raw metric data and a smoothed moving average 
  args:
    axs : artist
      axis to use for plt
    arr : ndarray
      array of loss data 
    idx : int
      which column to plot
    ylabel : str
    window_size : int 
      size of smoothing to apply
  returns
    None
  """
  
  ax.scatter(xdata,ydata,label="raw data")
  
  try:
    ax.plot(xdata[window_size:-window_size],movingaverage(ydata,window_size),linestyle="dashed",label=f"Moving average with size {window_size}")
  except IndexError: #error raised if num data points less than window size.  
    pass

  if "KL" in ylabel:
 
    ax.set_yscale("log")

  #set axis labels
  ax.set_xlabel("Epoch")
  ax.set_ylabel(ylabel)
  
  return 

def plot_loss_metrics(folder):

  fig,axs = plt.subplots(nrows=2,ncols=3)
  fig.set_figheight(15)
  fig.set_figwidth(25)

  epoch,Lhat,cost,dynamic_kl,static_kl,static_kl_from_ref,ref_var = np.loadtxt(folder+"/loss.csv",
                                                                               delimiter=",",skiprows=1,unpack=True)  

  for ax,metric,ylabel in zip(axs.flatten(),
                        [Lhat,cost,dynamic_kl,static_kl_from_ref,static_kl,ref_var],
                        [r"Objective, $\hat{\mathcal{L}}$","Action (Dynamic+Static KL)","Dynamic KL","Static KL from Reference","Static KL from Benchmark","Final Position Variance"]):

    plot_metric(ax,epoch,metric,ylabel)
  
  plt.tight_layout()
  fig.savefig(folder+"/"+"loss.png", bbox_inches='tight')
  
  plt.close()

  return


if __name__=="__main__":
  update_mpl()
  folder = fetch_results_folder_from_cmd()
  plot_loss_metrics(folder)
