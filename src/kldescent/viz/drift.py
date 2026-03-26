import numpy as np
import matplotlib.pyplot as plt

from kldescent.viz.plots import make_fig_subplots

def format_drift_ax(ax1,time):

    ax1.set_title(f"t={time:.4f}")
    ax1.set_xlabel("q")

def plot_drift(times,net_vals,benchmark_vals=None,err_vals =None, qmax = 5):
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
    qmax=2.5
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

        if err_vals is not None: 
            ax[t].fill_between(np.linspace(-qmax,qmax,1000),
                            net_vals[t] - 3*err_vals[t],
                            net_vals[t] + 3*err_vals[t],
                            label=r"90% confidence interval",
                            alpha=0.2)

    h,l = ax[0].get_legend_handles_labels()

    legend = fig.legend(h,l,
                frameon = False,
                handlelength = 1,
                ncols = 3)

    #move the legend
    legend.set_bbox_to_anchor(bbox=(0.95,0.07))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    return fig