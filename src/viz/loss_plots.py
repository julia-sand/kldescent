import os
import string

import numpy as np 

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt

from plots import PlotFuns

def plot_loss(folder):
  if folder == None:
    
    folder="tests/UnderdampedHalfGaussian"

  plotter = PlotFuns()
  plotter.setresultsdir(folder)
  plotter.plot_loss_metrics(folder+"/loss.csv")

if __name__=="__main__":
  from ..utils.parser import fetch_results_folder_from_cmd

  plot_loss(fetch_results_folder_from_cmd())
  #from ..sde.boundary import *
  #boundary_cond = UnderdampedHalfGaussian()
  #fileloc=f"tests/{boundary_cond.__class__.__name__}"
  #filename = fileloc+"/loss.csv" #add filename of loss.csv to plot
  #plotter = PlotFuns(boundary_cond)
  #plotter.setresultsdir(fileloc)
  #plotter.plot_loss_metrics(filename)