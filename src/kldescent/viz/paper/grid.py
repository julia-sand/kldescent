"""
Standalone script to generate a 5x5 grid comparison of final-time histograms
sweeping across epsilon and M parameter folders.

Usage:
Add all folders for plotting into top level results folder. Automatically formats 
grid based on the values of epsilon and M. The script will run agnostic of the folder naming 
scheme within the top level folder

Pass folder using --folder. 

If no --epoch argument is passed, automatically selects lowest loss for each example. 


Run as module python3 -m ...

Saves grid plot in folder. 
"""

import numpy as np
import matplotlib.pyplot as plt

# Framework imports
from kldescent.utils.parser import fetch_results_folder_from_cmd
from kldescent.utils.misc import list_subfolders

# Visualization imports
from kldescent.viz.plots import update_mpl, make_fig_subplots, get_all_subplot_handles_labels
from kldescent.viz.checkpoint_plots import PlotUtils
from kldescent.viz.distribution_plots import marginal_distribution

        
def generate_parameter_matrix_plot(axs, data):

    eps = np.empty(len(data))
    Ms = np.empty(len(data))

    for i in range(len(data)):
        utils_ = PlotUtils(folder=data[i]) #loads the data in the subfolder
        eps[i] = utils_.boundary_cond.params["epsilon"]
        Ms[i] = utils_.data_loaded["weight"]

        utils_.load_du_checkpoint()
        try:

            #if saved csv with samples exist, use that
            raise FileNotFoundError
        except FileNotFoundError:
            #if the file is not there
            x_du = utils_.integrator.update_ref_statistics(time=utils_.boundary_cond.params["Tf"],).cpu().numpy()
                
        marginal_distribution(axs[i],[x_du],utils_.boundary_cond.params["Tf"])

        axs[i].set_title(rf"$\varepsilon={eps[i]}$ | $M={Ms[i]}$\n(Epoch {utils_.epoch})", fontsize=20)

    return eps, Ms

def format_and_save(data,save_name):   

    update_mpl()

    #make figure object
    fig,axs = make_fig_subplots(ncols=len(data),nrows=1)
       
    axs = axs.flatten()

    eps, Ms = generate_parameter_matrix_plot(axs,data)


    eps_unique = np.unique(eps)
    Ms_unique = np.unique(Ms)

    positions = [(np.searchsorted(eps_unique, e), np.searchsorted(Ms_unique, m))
                for e, m in zip(eps, Ms)]
    
    resize_grid(fig, axs, len(eps_unique), len(Ms_unique), positions)

    #make legend
    handles, labels = get_all_subplot_handles_labels(fig)
    if handles:
        legend = fig.legend(handles, labels, ncols=4, frameon=False, handlelength=1.2, loc="upper center")
        legend.set_bbox_to_anchor((0.5, 0.02))
   
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
   
    fig.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"\nMatrix visualization successfully generated: {save_name}")


def resize_grid(fig, axs, nrows, ncols, positions):
    """Reassign existing axes into a new nrows x ncols grid.

    fig : Figure the axes belong to
    axs : sequence of Axes, already created (e.g. a flat 1xN row)
    nrows, ncols : shape of the new grid
    positions : sequence of (row, col) tuples, one per axes in axs
    """
    gs = fig.add_gridspec(nrows, ncols)
    for ax, (row, col) in zip(axs, positions):
        ax.set_subplotspec(gs[row, col])


if __name__ == "__main__":

    folder = fetch_results_folder_from_cmd()

    data = list_subfolders(folder)

    format_and_save(data,save_name=folder+"/final_time_matrix_comparison.png")
