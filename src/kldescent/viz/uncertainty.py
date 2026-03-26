import torch

from kldescent.viz.checkpoint_plots import PlotUtils
from kldescent.utils.parser import checkpoint_plot_params

@torch.no_grad
def compute_uncertainty_estimate(folders,epoch):

    time = torch.linspace(0,5,steps=20,requires_grad=False,dtype=torch.float32)

    net_output = []
    for folder in folders:

        plot_utils = PlotUtils(folder=folder,epoch=epoch,benchmark=None,wandb_run=None)
        plot_utils.load_du_checkpoint() 
        net_output.append(plot.utils.evaluate_net_output(plot_utils.net,time))
    



if __name__=="__main__":
    params = checkpoint_plot_params()
    epoch = params["epoch"]
    folders= ["f1","f2","f3","f4"]

    compute_uncertainty_estimate(folders,epoch)