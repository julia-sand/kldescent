import numpy as np
import matplotlib.pyplot as plt

from kldescent.viz.plots import make_fig_subplots


def format_drift_plot(axs,xaxis,fig,invert_labels):

    for ax in axs:
        if invert_labels:
            ax.set_xlabel("t")    
            ax.set_xticks([0,1,2,3,4,5])       
        else:
            ax.set_xlabel("q")

        ax.set_ylabel(r"$\mathbb{U}$")
        ax.set_ylim((-3,3))
        ax.set_xlim((np.min(xaxis),np.max(xaxis)))

    h,l = axs[0].get_legend_handles_labels()

    legend = fig.legend(h,l,
                frameon = False,
                handlelength = 1,
                ncols = 3)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

def plot_drift(times,net_vals,benchmark_vals=None,err_vals = None, 
                xaxis = np.linspace(-5,5,1000),
                invert_labels=False):
    """
    Plot drift at different times.

    Args:
        times (array-like): Array of time points.
        netvals (array-like): Values computed from the neural network.
        benchmarkvals (array-like, optional): Benchmark values to plot. If None,
            no benchmark curve is shown. Default is None.
        errbars (array-like, optional): Error bars for `netvals`. If None,
            no error bars are shown. Default is None.
        qmax (float, optional): Maximum value to plot on the q-axis. Default is 5.

    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """
    
    fig,ax = make_fig_subplots(nrows=1,ncols=len(times),squeeze=True)

    for t in range(len(times)):  

        if invert_labels:
            ax[t].set_title(f"q = {times[t]:.4f}")
        else:    
            ax[t].set_title(f"t = {times[t]:.4f}")

        ax[t].plot(xaxis,
                net_vals[t],
                label="N.N.")
                
        if benchmark_vals is not None:
            
            ax[t].plot(xaxis,
                        benchmark_vals[t],
                        label="Benchmark")
        
        #else:
        #    pass

        if err_vals is not None: 
            ax[t].fill_between(xaxis,
                            net_vals[t] - err_vals[t],
                            net_vals[t] + err_vals[t],
                            label=r"$\pm 1$ standard deviation",
                            alpha=0.2)
    
    format_drift_plot(ax,xaxis,fig,invert_labels)
    return fig

