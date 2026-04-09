import torch
import numpy as np
import matplotlib.pyplot as plt

from kldescent.viz.checkpoint_plots import PlotUtils
from kldescent.utils.parser import checkpoint_plot_params

from kldescent.viz.drift import plot_drift 

import pdb

@torch.no_grad
def plot_drift_uncertainty_errors(folders,epoch,numplots=5,invert=False,use_benchmark=True):

    if invert:
        times = [-1,-0.5,0,0.5,1]#np.sort(np.random.uniform(-5, 5,numplots))
        xaxis = np.linspace(0,5,1000)
    else:
        times = np.sort(np.random.uniform(0, 5,numplots))
        xaxis = np.linspace(-5,5,1000)

    net_vals = []
    err_vals = []
    for time in times:
        err_vals_temp,net_vals_temp = compute_uncertainty_estimate(time,folders,epoch,invert=invert)
        net_vals.append(net_vals_temp)
        err_vals.append(err_vals_temp)

    if use_benchmark:
        benchmark_vals = fetch_benchmark_sol(times,folders[0],epoch,invert)
    else: 
        benchmark_vals = None 

    fig = plot_drift(times,net_vals,benchmark_vals=benchmark_vals,err_vals = err_vals,
                        xaxis=xaxis,
                        invert_labels=invert)
    fig.savefig("test_gauss_invert_1.png")
    plt.close("all")

@torch.no_grad
def fetch_benchmark_sol(times,folder,epoch,invert):

    plot_utils = PlotUtils(folder=folder,epoch=epoch)

    plot_utils.load_benchmark_solution()

    if invert:
        return plot_utils.get_benchmark_drift_data_inverted(times)
    else:
        return plot_utils.get_benchmark_drift_data(times)

@torch.no_grad
def compute_uncertainty_estimate(time,folders,epoch,invert=False):
    #pdb.set_trace()
    net_output = []
    for folder in folders:

        plot_utils = PlotUtils(folder=folder,epoch=epoch,benchmark=None,wandb_run=None)
        plot_utils.load_du_checkpoint() 
        
        if invert:
            net_output.append(plot_utils.evaluate_net_output_inverted(plot_utils.net,time))  
        else: 
            net_output.append(plot_utils.evaluate_net_output(plot_utils.net,time))  

    std = np.std(net_output, axis=0)

    return std.flatten(),net_output[-3].flatten()

if __name__=="__main__":
    params = checkpoint_plot_params()
    epoch = 450
    folders = []
    for i in range(1,20):
        print(i)
        folders.append(f"../../../../../scratch/project_2011332/kldescent_gauss_seed_3/SEED_{i}")#kldescent_seed/SEED_{i}")
    
    plot_drift_uncertainty_errors(folders,epoch,invert=True)