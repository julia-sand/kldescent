"""Create the contour plots showing the full joint pdf, where the pdf is computed 
either using a neural network interpolation, or using a Monte Carlo integration"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import torch

from kldescent.sde.girsanovpdf import compute_log_pdf
from kldescent.viz.plots import *
from kldescent.utils.misc import append_array_to_csv

@torch.no_grad
def generate_mesh(x_samples,qmax=6):
    """
    Generate a mesh in two-dimensional phase space.

    Parameters
    ----------
    x_samples : int
        Number of samples in the space discretisation.
    qmax : float, optional
        Maximum value of the axis, symmetric in positive and negative. Default is 6.
    """

    p_init = torch.linspace(-qmax,qmax,x_samples, requires_grad=False,dtype=torch.float32)
    #q_init = torch.linspace(-qmax,qmax,x_samples, requires_grad=False,dtype=torch.float32)
        
    return torch.meshgrid(p_init,p_init,indexing='ij')

@torch.no_grad
def compute_rho_girsanov(x_samples=40, integrator=None, mc_samples=64, Tf=None, **kwargs):
    """
    Compute an estimate of the final joint distribution using Girsanov's theorem.

    Parameters
    ----------
    x_samples : int, optional
        Number of space discretisation steps. Default is 40.
    integrator : SDEBase subclass instance
        Initialised integrator providing net, boundary, and sde_params.
    mc_samples : int, optional
        Number of Monte Carlo samples. Default is 64.
    Tf : float, optional
        Final time for the plot (unused; inherited from integrator boundary).

    Returns
    -------
    xout : ndarray
        2D mesh of points of the state space.
    jointout : ndarray
        Estimated joint distribution using Girsanov's theorem.
    """

    Q, P = generate_mesh(x_samples, qmax=4)
    xinit = torch.stack((P, Q), dim=-1).reshape(-1, 2)

    with torch.no_grad():
        integrator.net.eval()
        log_pdf = compute_log_pdf(xinit.clone(), integrator, mc_samples=mc_samples, Tf=Tf)
        pdf_out = torch.exp(log_pdf).cpu().numpy()

    return xinit.cpu().numpy(), pdf_out

@torch.no_grad
def distribution_plot(plot_loc,model,Tf,epoch=1,qmax=5):

    #compute marginals 
    xx, yy = torch.meshgrid(torch.linspace(-qmax,qmax, 100), torch.linspace(-qmax,qmax, 100), indexing="xy")
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    
    #find probabilities 
    pdf_temp = torch.exp(model.log_prob(zz))
    pdf_temp[torch.isnan(pdf_temp)] = 0

    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(25)

    xout,jointout = zz,pdf_temp.detach().clone().numpy().flatten()

    plot_joint_panel(plt.gca(),5,jointout,xout)

    fig.tight_layout()
    fig.savefig(plot_loc+f"/contour_distribution_{epoch}.png",bbox_inches="tight")        

    plt.close()

    fig1 = plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, pdf_temp.view(*xx.shape).detach().clone().numpy(), cmap='coolwarm')
    fig1.savefig(plot_loc+"/pdf_distribution.png",bbox_inches="tight") 
    plt.close()

def get_norm(x,y):
    idx = np.argsort(x)
    return np.trapezoid(y[idx],x[idx])

def add_boundary_conditions_scatter_panel(ax_pmarginal,ax_qmarginal,time,boundary):

    if time==0:
        #ax_qmarginal.fill_between(
        #        boundary.q_marginal_initial_np(np.linspace(-boundary.qmax,boundary.qmax,)),np.linspace(-boundary.qmax,boundary.qmax,),color="orange",label="Assigned Initial Distribution")

        ax_qmarginal.fill_betweenx(np.linspace(-boundary.qmax,boundary.qmax,),
                boundary.q_marginal_initial_np(np.linspace(-boundary.qmax,boundary.qmax,)),color="orange",label="Assigned Initial Distribution")
        ax_pmarginal.fill_between(np.linspace(-boundary.qmax,boundary.qmax,),
                normal_distribution(np.linspace(-boundary.qmax,boundary.qmax,),0,1),color="orange",label="Assigned Initial Distribution")

    elif time==boundary.params["Tf"]:
        #ax_qmarginal.fill_between(
        #        boundary.q_marginal_final_np(np.linspace(-boundary.qmax,boundary.qmax,)),np.linspace(-boundary.qmax,boundary.qmax,),color="orange",label="Target Final Distribution")
        ax_qmarginal.fill_betweenx(np.linspace(-boundary.qmax,boundary.qmax,),
        boundary.q_marginal_final_np(np.linspace(-boundary.qmax,boundary.qmax,)),color="orange",label="Target Final Distribution")
        ax_pmarginal.fill_between(np.linspace(-boundary.qmax,boundary.qmax,),
                normal_distribution(np.linspace(-boundary.qmax,boundary.qmax,),0,1),color="darkorange",label="Target Final Distribution")

    else: 
        pass

def plot_marginal_joint(ax,x0_marginal,x0_init):

    ax.plot(x0_init,x0_marginal
                        ,label="Girsanov")


def scatter_panel_layout(ax,time,boundary=None):
    """format axes for the contour plot and add assigned boundary conditions"""

    #make inset axes to show marginal distributions
    ax_pmarginal = ax.inset_axes([0, 1.05, 1, 0.6])
    ax_qmarginal = ax.inset_axes([-0.65, 0, 0.6, 1])
    ax_qmarginal.invert_xaxis()

    ###ticks formatting and labelling
    ax_pmarginal.yaxis.tick_right()
    ax.yaxis.tick_right()

    if boundary is not None:  
        plot_title_value = round(time/boundary.params["Tf"],2)
        ax.set_title(f"t={plot_title_value}Tf")
        
        #set x axis limits
        ax.set_ylim((-boundary.qmax,boundary.qmax))
        ax.set_xlim((-boundary.qmax,boundary.qmax))
        ax_qmarginal.set_ylim((-boundary.qmax,boundary.qmax))
        ax_pmarginal.set_xlim((-boundary.qmax,boundary.qmax))

    else: 
        ax.set_title(f"t={time}")


    ax_pmarginal.tick_params(
        axis='y',
        which='both',
        length=5,
        rotation=-20,
        pad = -3)

    ax_qmarginal.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',
        rotation=70,
        length =5,
        pad = -3,
        labelleft=False)

    ax_qmarginal.tick_params(
        axis='y',          # changes apply to the y-axis
        which='both',      #both major and minor ticks
        left=False,
        labelleft=False)

    ax_pmarginal.set_xticklabels([])

    ax.set_xlabel(r"$p$")
    ax_qmarginal.set_xlabel(r"$\rho_t(q)$")
    ax_qmarginal.set_xlim(right=-0.01)#,left=0.6)

    ax.set_ylabel(r"$q$")
    ax_pmarginal.set_ylabel(r"$\rho_t(p)$")
    ax.yaxis.set_label_position("right")
    ax_pmarginal.set_ylim(bottom=-0.01,top=0.6)

    ax.set_aspect('equal', adjustable='box')
    #ax.set_xticks([-2.5,0,2.5])

    if boundary is not None:
        add_boundary_conditions_scatter_panel(ax_pmarginal,ax_qmarginal,time,boundary)

    return ax, ax_pmarginal,ax_qmarginal

def _dedupe_axis_values(values, tol=1e-10):
    """Return the unique sorted values of a 1D axis.

    For a tensor-product grid the same axis value repeats across every
    cross-section, so the number of distinct entries equals the grid size
    along that axis.
    """
    values = np.asarray(values, dtype=float).ravel()
    order = np.argsort(values)
    vals = values[order]
    keep = np.concatenate(([True], np.diff(vals) > tol))
    return vals[keep]

def downsample_to_square(grid, nq, np_):
    """Axis-stride subsample a rectangular (nq, np_) grid to a square (M, M).

    Method 1: keep every k-th grid line along the over-sampled axis so both
    axes have length M = min(nq, np_). No interpolation is performed; existing
    values are simply selected on the retained grid lines.

    Parameters
    ----------
    grid : np.ndarray
        Joint distribution on a rectangular grid, shape (nq, np_).
    nq, np_ : int
        Number of grid lines along q (rows) and p (columns). Must match
        ``grid.shape``.

    Returns
    -------
    grid_s : np.ndarray
        Square joint grid of shape (M, M).
    sq, sp : np.ndarray
        Integer index arrays used to select rows (q) and columns (p).
    """
    nq, np_ = grid.shape
    M = min(nq, np_)
    sq = np.linspace(0, nq - 1, M).round().astype(int)
    sp = np.linspace(0, np_ - 1, M).round().astype(int)
    return grid[np.ix_(sq, sp)], sq, sp

def reshape_joint(joint_out, xs=None):
    """Reshape a flattened joint distribution to its (possibly square) grid.

    For a dense tensor-product grid the distinct axis values in ``xs`` give
    the rectangular shape (nq, np_). If the grid is not square it is
    downsampled to M x M (M = min(nq, np_)) via axis-stride subsampling so
    the downstream trapezoid marginals work on a uniform square grid.

    If ``xs`` is None the legacy square reshape is used as a fallback.

    Parameters
    ----------
    joint_out : np.ndarray
        Flattened joint distribution.
    xs : np.ndarray, optional
        Input coordinates of shape (N, 2), each row a (q, p) pair.

    Returns
    -------
    np.ndarray
        Square joint grid (M, M).
    """
    joint_out = np.nan_to_num(np.asarray(joint_out, dtype=float).ravel(),
                              nan=0.0, posinf=0.0, neginf=0.0)
    if xs is None:
        n = joint_out.size
        side = int(round(np.sqrt(n)))
        if side * side != n:
            raise ValueError(
                f"joint_out has {n} points and cannot be reshaped to a square; "
                f"pass xs to enable rectangular-grid downsampling.")
        return joint_out.reshape((side, side))

    xs = np.asarray(xs)
    q_vals = _dedupe_axis_values(xs[:, 0])
    p_vals = _dedupe_axis_values(xs[:, 1])
    if q_vals.size * p_vals.size != joint_out.size:
        raise ValueError(
            f"Data does not form a dense tensor-product grid: "
            f"nq*np_ = {q_vals.size}*{p_vals.size} but joint_out has "
            f"{joint_out.size} points.")
    grid = joint_out.reshape((q_vals.size, p_vals.size))
    if grid.shape[0] == grid.shape[1]:
        return grid
    grid_s, _, _ = downsample_to_square(grid, grid.shape[0], grid.shape[1])
    return grid_s


def compute_marginals_from_joint(xs,joint_out):
    """
    Compute normalized position (q) and momentum (p) marginals from a joint distribution.

    Parameters
    ----------
    xs : ndarray
        Grid or coordinate values corresponding to the joint distribution.
    joint_out : ndarray
        Joint distribution data.

    Returns
    -------
    pinit : ndarray
        Coordinate values along each axis.
    qmargnorm : ndarray
        Normalized position marginal distribution.
    pmargnorm : ndarray
        Normalized momentum marginal distribution.
    normall : float
        Total normalisation constant.
    """

    # Build the (possibly downsampled) square joint grid and the matching
    # square axis values. `xs` is expected to be a dense tensor-product grid
    # so the distinct values on each axis give the original (nq, np_) shape.
    joint_out_temp = reshape_joint(joint_out, xs)
    sq, sp = joint_out_temp.shape

    xs_arr = np.asarray(xs)
    q_axis_full = _dedupe_axis_values(xs_arr[:, 0])
    p_axis_full = _dedupe_axis_values(xs_arr[:, 1])

    if joint_out_temp.shape[0] != q_axis_full.size or joint_out_temp.shape[1] != p_axis_full.size:
        # grid was downsampled to (sq, sp); select the matching square axis values
        q_axis = q_axis_full[np.linspace(0, q_axis_full.size - 1, sq).round().astype(int)]
        p_axis = p_axis_full[np.linspace(0, p_axis_full.size - 1, sp).round().astype(int)]
    else:
        q_axis = q_axis_full
        p_axis = p_axis_full

    #compute marginals
    qmarg = np.trapezoid(joint_out_temp, x = p_axis, axis=1)
    pmarg = np.trapezoid(joint_out_temp, x = p_axis, axis=0)

    #compute normalisation
    qnorm = get_norm(p_axis, qmarg)
    pnorm = get_norm(p_axis, pmarg)

    return p_axis, qmarg/qnorm, pmarg/pnorm, qnorm*pnorm

def _grid_marginals(q, p, values, n_bins=64):
    """Compute normalised 1-D marginals from scattered (q, p) points.

    Works without assuming the points sit on a dense tensor-product grid:
    each point simply contributes its ``values`` weight to the bin it falls
    into along the axis being marginalised over.

    Parameters
    ----------
    q, p : ndarray
        Coordinate and momentum values for each point (same length).
    values : ndarray
        Per-point joint weight (the ``joint_out`` value at each point).
    n_bins : int
        Number of bins along each axis.

    Returns
    -------
    q_axis, qmarg_norm, pmarg_norm, normall
        ``q_axis`` is the q bin centres (used as the p-scaling reference,
        matching the grid-based path), ``qmarg_norm``/``pmarg_norm`` are the
        L1-normalised position/momentum marginals, and ``normall`` is the
        total integrated joint weight (``qnorm * pnorm``).
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    values = np.asarray(values, dtype=float)

    # Equal-width bins spanning each axis, clamped so out-of-range points
    # land in the nearest edge bin.
    q_edges = np.linspace(q.min(), q.max(), n_bins + 1)
    p_edges = np.linspace(p.min(), p.max(), n_bins + 1)

    qmarg = np.bincount(np.clip(np.digitize(q, q_edges[1:-1], right=True), 0, n_bins - 1),
                        weights=values)
    pmarg = np.bincount(np.clip(np.digitize(p, p_edges[1:-1], right=True), 0, n_bins - 1),
                        weights=values)

    # L1 normalise to unit area so the marginals read as densities.
    qnorm = np.sum(qmarg) * (q_edges[1] - q_edges[0])
    pnorm = np.sum(pmarg) * (p_edges[1] - p_edges[0])
    qnorm = qnorm if qnorm > 0 else 1.0
    pnorm = pnorm if pnorm > 0 else 1.0

    q_axis = 0.5 * (q_edges[1:] + q_edges[:-1])
    return q_axis, qmarg / qnorm, pmarg / pnorm, qnorm * pnorm


def joint_distributions_scatter_panel(
                                    ax,
                                    ax_pmarginal,
                                    ax_qmarginal,
                                    joint_out,
                                    xs,time,vmax=0.5,
                                    benchmark=None):
    """
    Plot a single panel of a joint distribution.

    The central plot visualizes the joint distribution, while the marginal
    distributions are estimated from the same data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the joint distribution.
    ax_pmarginal : matplotlib.axes.Axes
        The axes for the marginal distribution of p.
    ax_qmarginal : matplotlib.axes.Axes
        The axes for the marginal distribution of q.
    joint_out : np.ndarray
        Flattened joint distribution of shape ``(n**2,)``.
    xs : np.ndarray
        Input coordinates of shape ``(n**2, 2)``, where each row is a (q, p) pair.
    time : float
        Time of the evolution.
    vmax : float, optional
        Upper limit for the color scale. Default is 0.5.
    benchmark : object, optional
        Benchmark solution handler. If None, benchmark values are not plotted.
    """

    p_init_out,qmarg_norm,pmarg_norm,normall = compute_marginals_from_joint(xs,joint_out)

    # Build the (possibly downsampled) square joint grid and matching square
    # axis values so that the per-point scatter colours are looked up on the
    # same grid used for the marginals.
    joint_out_temp = reshape_joint(joint_out, xs)
    sq, sp = joint_out_temp.shape

    xs_arr = np.asarray(xs)
    q_axis_full = _dedupe_axis_values(xs_arr[:, 0])
    p_axis_full = _dedupe_axis_values(xs_arr[:, 1])

    if sq == q_axis_full.size and sp == p_axis_full.size:
        q_axis = q_axis_full
        p_axis = p_axis_full
        # Original (q, p) coordinates map directly to grid (i, j).
        i_ds = np.searchsorted(q_axis, xs_arr[:, 0])
        j_ds = np.searchsorted(p_axis, xs_arr[:, 1])
        i_ds = np.clip(i_ds, 0, sq - 1)
        j_ds = np.clip(j_ds, 0, sp - 1)
    else:
        sq_idx = np.linspace(0, q_axis_full.size - 1, sq).round().astype(int)
        sp_idx = np.linspace(0, p_axis_full.size - 1, sp).round().astype(int)
        q_axis = q_axis_full[sq_idx]
        p_axis = p_axis_full[sp_idx]
        # For each original point, find the nearest downsampled axis index so
        # the colour is taken from the retained grid line.
        i_ds = np.searchsorted(q_axis, xs_arr[:, 0], side="right") - 1
        j_ds = np.searchsorted(p_axis, xs_arr[:, 1], side="right") - 1
        # Handle exact match at the first element (searchsorted gives 0).
        i_ds = np.where(i_ds < 0, 0, i_ds)
        j_ds = np.where(j_ds < 0, 0, j_ds)
        i_ds = np.clip(i_ds, 0, sq - 1)
        j_ds = np.clip(j_ds, 0, sp - 1)
    c_vals = joint_out_temp[i_ds, j_ds]

    #plot and format marginals
    plot_marginal_joint(ax_qmarginal, p_axis, qmarg_norm)
    plot_marginal_joint(ax_pmarginal, pmarg_norm, p_axis)

    if benchmark is not None:
        #plot marginals
        x,ptx = benchmark.get_benchmark_quantity(np.round(time,6),quantity = "underdamped marginal",data="perturbative underdamped",time_type="t0")
        ax_qmarginal.plot(ptx,x,color="gray",linestyle="dashed",label="Benchmark Predicted")

    #plot joint distribution
    ax.scatter(xs_arr[...,0],xs_arr[...,1]/normall,c=c_vals
                ,cmap="Blues",norm="log",vmin=0.0001,vmax=vmax)

def make_colourbar(fig,cbar_ax,vmax):
    """add color bar to fig"""

    pos = cbar_ax.get_position()

    sm =  ScalarMappable(norm="log",cmap=plt.get_cmap("Blues"))
    sm.set_array([])
    sm.set_clim(vmin=0.001, vmax=vmax)

    cb = fig.colorbar(sm,
                        cax=cbar_ax,
                        orientation="horizontal")

    cbar_ax.set_xlabel("Joint Distribution")
    cbar_ax.set_position([0.25, 0.1, 0.5, 0.02])

def plot_joint_panel(ax,time,jointout,xout,boundary,benchmark=None):
    
    ax, ax_pmarginal,ax_qmarginal = scatter_panel_layout(ax,time,boundary)

    joint_distributions_scatter_panel(ax,
                                            ax_pmarginal,
                                            ax_qmarginal,
                                            jointout,
                                            xout,time,vmax=0.5,benchmark=benchmark)

def write_csv_header_joint(csv_path):

    # write header once
    with open(csv_path, "w") as f:
        f.write("t,q,p,joint_out\n")

def save_joint_to_csv(t,xinit,jointout,filename):

    # vectorized assembly
    t_col = np.full((xinit.shape[0], 1), t)
    append_array_to_csv(filename,np.hstack([
                                            t_col,
                                            xinit,
                                            jointout[:, None].reshape(-1,1),
                                        ]))
    
    return 

def contour_plot_(axs, integrator, boundary,
                            resultsdir,
                            plottimes = [0,1,2,3,4,5],
                            benchmark=None,
                            mc_samples=64):
    
    
    #initialise the csv
    filename = resultsdir+"/joint_distribution.csv"
        
    # ==========================================================
    # USE EXISTING CSV
    # ==========================================================
    if os.path.exists(filename):

        loaded = load_contour_csv(filename)

        for i, t in enumerate(plottimes):

            if t not in loaded:
                continue

            plot_joint_panel(
                axs[i],
                t,
                loaded[t]["jointout"],
                loaded[t]["xinit"],
                boundary=boundary,
                benchmark=benchmark,
            )

        print(f"Loaded contour data from: {filename}")
        return

    write_csv_header_joint(filename)

    #axs = make_gridspec_subplots(fig,gs_top)
   
    Q,P = generate_mesh(x_samples=20,qmax=4)
    xinit = torch.stack((Q, P), dim=-1).reshape(-1, 2)
    

    for i in range(len(plottimes)):

        jointout = torch.exp(compute_log_pdf(xinit.clone(), integrator, mc_samples=mc_samples, Tf=plottimes[i]))
            
        plot_joint_panel(axs[i],plottimes[i],jointout.cpu().numpy(),xinit.cpu().numpy(),
                            boundary = boundary,
                            benchmark = benchmark)
        
        save_joint_to_csv(plottimes[i],xinit.cpu().numpy(),jointout.cpu().numpy(),filename)

def make_axes_contour_plot(len_plottimes):

    fig = plt.figure(figsize=(20, 15), constrained_layout=False)

    # Outer grid: 2 rows (plots + colorbar + legend)
    gs = fig.add_gridspec(nrows=3,
                          ncols=1,
                            left=0.1, right=1,
                            bottom=0, top=0.95,
                            hspace=0.3,
                          height_ratios=[10, 0.5, 1])                                    

    ncols_top=np.maximum(2,len_plottimes//2)
    nrows_top=2

    # ---- TOP: nested subplots grid ----
    gs_top = gs[0].subgridspec(nrows_top, ncols_top,
                                wspace=0.5)
    
    axs = make_gridspec_subplots(fig, gs_top, ncols_top,nrows_top)
    
    # ---- BOTTOM: colorbar axis ----
    cax = fig.add_subplot(gs[1])
    make_colourbar(fig,cax,vmax=0.5)

    #add legend
    lax = fig.add_subplot(gs[2])
    lax.axis("off")  # hide legend axis box

    return fig,axs,lax 

def add_legend_contour_plot(fig,lax):

    h,l = get_all_subplot_handles_labels(fig)
    make_legend(lax,h,l)

    fig.subplots_adjust(hspace=1.1,wspace=1,bottom=0.18)
    

def make_contour_plot_(integrator, boundary,
                            resultsdir,
                            plottimes = [0,1,2,3,4,5],
                            benchmark=None,
                            mc_samples=64):

    fig, axs, lax = make_axes_contour_plot(len(plottimes))

    contour_plot_(axs, integrator, boundary,
                            resultsdir,
                            plottimes,
                            benchmark,
                            mc_samples=mc_samples)

    add_legend_contour_plot(fig, lax)

    save_fig_(fig, "contour_distribution", loc=resultsdir)

    plt.close()

def load_contour_csv(filename):
    """
    Load CSV produced by contour_plot_ and group by time.

    Returns
    -------
    data_dict : dict

        {
            t0: {
                "xinit": ndarray shape (N,2),
                "jointout": ndarray shape (N,)
            },
            t1: ...
        }
        dictionary of values xinit and probability distribution, with t as the key
    """

    # columns: t,q,p,joint_out
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    t_vals = data[:, 0]
    q_vals = data[:, 1]
    p_vals = data[:, 2]
    joint_vals = data[:, 3]

    unique_times = np.unique(t_vals)

    data_dict = {}

    for t in unique_times:

        mask = t_vals == t

        xinit = np.column_stack([
            q_vals[mask],
            p_vals[mask],
        ])

        jointout = joint_vals[mask]

        data_dict[t] = {
            "xinit": xinit,
            "jointout": jointout,
        }

    return data_dict

def make_contour_plot_from_csv(folder,boundary=None,benchmark=None):

    loaded = load_contour_csv(folder+"/joint_distribution.csv")

    fig,axs,lax = make_axes_contour_plot(len(loaded))
    
    for i, t in enumerate(sorted(loaded.keys())):

        plot_joint_panel(
            axs[i],
            t,
            loaded[t]["jointout"],
            loaded[t]["xinit"],
            boundary=boundary,
            benchmark=benchmark,
        )

    add_legend_contour_plot(fig,lax)
    
    save_fig_(fig,"contour_distribution",loc=folder)
    
    plt.close()


    print(f"Loaded contour data from: {folder}")
    return

if __name__=="__main__":
    from kldescent.utils.parser import fetch_results_folder_from_cmd
    from kldescent.viz.plots import update_mpl

    update_mpl()
    folder=fetch_results_folder_from_cmd()
    
    make_contour_plot_from_csv(folder,)