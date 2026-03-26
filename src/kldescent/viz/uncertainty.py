import torch
import numpy as np
import matplotlib.pyplot as plt

from kldescent.viz.checkpoint_plots import PlotUtils
from kldescent.utils.parser import checkpoint_plot_params

from kldescent.viz.drift import plot_drift 

import pdb

@torch.no_grad
def plot_drift_uncertainty_errors(folders,epoch,numplots=5):

    times = np.sort(np.random.uniform(0, 5,numplots))

    net_vals = []
    err_vals = []
    for time in times:
        err_vals_temp,net_vals_temp = compute_uncertainty_estimate(time,folders,epoch)
        net_vals.append(net_vals_temp)
        err_vals.append(err_vals_temp)

    fig = plot_drift(times,net_vals,err_vals = err_vals)
    fig.savefig("test.png")
    plt.close("all")

@torch.no_grad
def compute_uncertainty_estimate(time,folders,epoch):
    #pdb.set_trace()
    net_output = []
    for folder in folders:

        plot_utils = PlotUtils(folder=folder,epoch=epoch,benchmark=None,wandb_run=None)
        plot_utils.load_du_checkpoint() 
        
        net_output.append(plot_utils.evaluate_net_output(plot_utils.net,time))  

    std = np.std(net_output, axis=0)

    return std.flatten(),net_output[-1].flatten()

if __name__=="__main__":
    params = checkpoint_plot_params()
    epoch = 60
    folders = []
    for i in range(1,11):
        print(i)
        folders.append(f"../../../../../scratch/project_2011332/kldescent_seed/SEED_{i}")
    
    plot_drift_uncertainty_errors(folders,epoch)