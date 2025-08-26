import os
import string

from plots import PlotFuns
from ..utils.parser import fetch_results_folder_from_cmd

def plot_loss(folder):
  if folder == None:
    
    folder="tests/UnderdampedHalfGaussian"

  plotter = PlotFuns()
  plotter.setresultsdir(folder)
  plotter.plot_loss_metrics(folder+"/loss.csv")

if __name__=="__main__":

  plot_loss(fetch_results_folder_from_cmd())
  