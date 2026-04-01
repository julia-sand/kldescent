import numpy as np
import matplotlib.pyplot as plt

from kldescent.viz.plots import make_fig_subplots


def format_drift_plot(axs,fig,invert_labels):

    for ax in axs:
        if invert_labels:
            ax.set_xlabel("t")           
        else:
            ax.set_xlabel("q")

        ax.set_ylabel(r"$\mathbb{U}$")
        #ax.set_ylim((-5,5))
        #ax.set_xlim((-1.5,1.5))

    h,l = axs[0].get_legend_handles_labels()

    legend = fig.legend(h,l,
                frameon = False,
                handlelength = 1,
                ncols = 3)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

def plot_drift(times,net_vals,benchmark_vals=None,err_vals =None, 
                xaxis = np.linspace(-5,5,1000),
                invert_labels=False):
    """Make plot of drift at difference times
    args:
        times : arr
            number of t
        netvals : arr
            evaluated neural network values
        benchmarkvals : arr
            benchmark values to plot. Default None, in which case none are plotted.
        errbars : arr
            error bars to put on the netvals. Default None, in which case no error bars 
        qmax : float
            Default 5, maximum value to plot. 
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
                        label="Benchmark N.N.")
        
        else:
            pass

        if err_vals is not None: 
            ax[t].fill_between(xaxis,
                            net_vals[t] - 3*err_vals[t],
                            net_vals[t] + 3*err_vals[t],
                            label=r"90% confidence interval",
                            alpha=0.2)
    format_drift_plot(ax,fig,invert_labels)
    return fig

