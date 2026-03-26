import numpy as np
import matplotlib.pyplot as plt

from kldescent.viz.plots import make_fig_subplots

def format_drift_ax(ax1,time):

    ax1.set_title(f"t={time:.4f}")
    ax1.set_xlabel("q")

def plot_drift(times,net_vals,benchmark_vals,qmax = 5):

    fig,ax = make_fig_subplots(nrows=1,ncols=len(times),squeeze=True)

    for t in range(len(times)):  
        
        ax[t].plot(np.linspace(-qmax,qmax,1000),
                net_vals[t],
                label="N.N.")
        
        format_drift_ax(ax[t],times[t])

        ax[t].set_ylabel(r"$\mathbb{U}$")
        #ax[t].set_ylim((-5,5))
        
        if benchmark_vals is not None:
            
            ax[t].plot(np.linspace(-qmax,qmax,1000),
                benchmark_vals[t],
                label="Benchmark N.N.")
        
        else:
            pass
            #print("No reference solution was given. Plots will not include reference values.")

    h,l = ax[0].get_legend_handles_labels()
    #plt.legend()
    legend = fig.legend(h,l,
                frameon = False,
                handlelength = 1,
                ncols = 2)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    return fig