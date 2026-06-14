"""Plots for the benchmark solution computed with perturbation theory"""
import matplotlib.pyplot as plt

from kldescent.viz.drift import plot_drift
from kldescent.viz.plots import update_mpl
from kldescent.benchmark.perturbative.driftinterp import benchmark_evaluate_net

def plot_benchmark_drift(benchmark_):
    
    update_mpl()

    times = [0,1.5,3,5]
    x = np.unique(benchmark_.get_benchmark_quantity(times[0], "underdamped drift", data="perturbative underdamped",time_type="t2")[0])

    net_vals = []
    benchmark_vals = []
    
    for time in times:
        
        net_vals.append(benchmark_evaluate_net(benchmark_.du_net_benchmark,x,time))
        #benchmark_vals.append((-benchmark_.boundary.params["epsilon"]/2**0.5)*benchmark_.get_benchmark_quantity(time, "underdamped drift", data="perturbative underdamped")[1])
        
    fig = plot_drift(times,net_vals,
                        benchmark_vals=None,
                        xaxis = x)
    
    for ax in fig.get_axes():
        ax.set_xlim((-5,5))
        ax.set_ylim((-20,20))   

    fig.savefig(benchmark_.folder_name+f"/perturbative_drift_trained.png",bbox_inches="tight")        

def plot_data(benchmark_,quantity,data,numplots=5):
    
    t_vec = benchmark_.get_timegrid()
    idx = np.random.randint(low=0,high=len(t_vec),size=5)
    times = np.sort(t_vec[idx])
    
    fig,axs = plt.subplots(ncols=len(times),nrows=1)
    fig.set_figheight(15)
    fig.set_figwidth(25)

    eps = benchmark_.boundary.params["epsilon"]
    for time in enumerate(times):
        q, ptq = benchmark_.get_benchmark_quantity(time[1], quantity, data=data,time_type="t0")
        axs[time[0]].plot(q,ptq,label="Benchmark data")
    
        axs[time[0]].set_title(f"t={time[1]}")

        axs[time[0]].set_xlim((-5,5))
        if quantity == "underdamped drift":
            du_net = -(2**0.5/eps)*benchmark_evaluate_net(benchmark_.du_net_benchmark,q,time[1])
            axs[time[0]].plot(q,du_net, label="NN interpolation")

    h,l = plt.gca().get_legend_handles_labels()
    #fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    fig.legend(h,l)

    fig.savefig(benchmark_.folder_name+f"/{data}_{quantity}.png",bbox_inches="tight")        

if __name__=="__main__":
    from kldescent.sde.boundary import *
    from kldescent.utils.parser import fetch_results_folder_from_cmd
    from kldescent.utils.parser import overdamped_argparser
    from kldescent.benchmark import PerturbativeBenchmarkSolution

    update_mpl()

    bridgetype = overdamped_argparser()["bridgetype"]

    boundary = UnderdampedLandauer()
    benchmark_ = PerturbativeBenchmarkSolution(bridge_type=bridgetype,boundary=boundary,
                        folder_name=fetch_results_folder_from_cmd())
    
    plot_data(benchmark_,quantity="value",data="overdamped")
    plot_data(benchmark_,quantity="marginal",data="overdamped")
    plot_data(benchmark_,quantity="marginal",data="perturbative underdamped")
    plot_data(benchmark_,quantity="underdamped drift",data="perturbative underdamped")
    plot_benchmark_drift(benchmark_)
