#import math
import torch
import numpy as np
#from scipy.stats import bootstrap
#import pdb

from ..model.nn import NeuralNetwork
from .plots import PlotFuns

def save_cumulants(save_array,x):
    #pdb.set_trace()
    temp = x.numpy()
    return np.append(save_array,np.array([[np.nanmean(temp[...,0]),
                                            np.nanmean(temp[...,1]),
                                            np.nanvar(temp[...,0]),
                                            np.nanvar(temp[...,1]),
                                            np.correlate(temp[...,0],temp[...,1])[0]/len(temp[...,0])]]),axis=0)#cross corr


def get_var_confidence_intervals(x):
    res = bootstrap(x, np.nanvar, confidence_level=0.9) 
    ci = res.confidence_interval
    return ci[0],ci[1]
 
def get_mean_confidence_intervals(x):
    res = bootstrap(x, np.nanmean, confidence_level=0.9) 
    ci = res.confidence_interval
    return np.array([[ci[0],ci[1]]])


def compute_ci(posmean_ci,
                posvar_ci,
                mommean_ci,
                momvar_ci,x):
    #temp = x.numpy()

    #posmean_ci = np.append(posmean_ci,get_mean_confidence_intervals(temp[...,[0]]),axis=0)
    #mommean_ci = np.append(mommean_ci,get_mean_confidence_intervals(temp[...,[1]]),axis=0)
    #posvar_ci = np.append(mommean_ci,get_var_confidence_intervals(temp[...,[0]]),axis=0)
    #momvar_ci = np.append(mommean_ci,get_var_confidence_intervals(temp[...,[1]]),axis=0)
    return  posmean_ci,posvar_ci,mommean_ci,momvar_ci

def compute_sde(net,
                marginal_plot_time,
                testing_samples,
                plotter,
                boundary):

    """
    Computes the SDE and returns plots 

    ====
    Parameters
    - net: neural network
    ====

    """
        
    t_vec = boundary.get_reference_timegrid(marginal_plot_time)

    #initialise x values at \delta_0 (i.e. x=0)
    xinit = torch.tensor(boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32)
    xref = torch.clone(xinit)

    #save the cumulants of the evolutions
    cumulants_nn = save_cumulants(np.empty((0,5)),xinit)
    cumulants_ref = save_cumulants(np.empty((0,5)),xref)
    posmean_ci = np.empty((0,2))
    posvar_ci = np.empty((0,2))
    mommean_ci = np.empty((0,2))
    momvar_ci = np.empty((0,2))
    
    with torch.no_grad():

        #self.generate_time_grid()
        for i in range(0,len(t_vec)-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = np.sqrt(dt)*torch.randn((testing_samples,1))
            du = net.forward(torch.column_stack((t_vec[i]*torch.ones(testing_samples,dtype=torch.float32)
                                                    ,xinit[:,0,None])))

            
            xinit = boundary.euler_maruyama_step(xinit,du,dt,dw,t_vec[i])
            xref = boundary.euler_maruyama_step(xref,boundary_cond.du_ref(xref[:,[0]],t_vec[i]),dt,dw,t_vec[i])
            cumulants_nn = save_cumulants(cumulants_nn,xinit)
            cumulants_ref = save_cumulants(cumulants_ref,xref)
            posmean_ci,posvar_ci,mommean_ci,momvar_ci = compute_ci(posmean_ci,
                                                                    posvar_ci,
                                                                    mommean_ci,
                                                                    momvar_ci,xinit) 

    plotter.marginal_distribution([xinit.numpy(),xref.numpy()],marginal_plot_time)
    plotter.plot_cumulants(t_vec,cumulants_nn,cumulants_ref,
                        [posmean_ci,posvar_ci,mommean_ci,momvar_ci])
    
    #free up the tensors
    del xinit 
    del du

    return 

def load_net(boundary_cond,epoch,folder=None):

    if folder == None:
        folder = "tests/"+f"{boundary_cond.__class__.__name__}"
    else: #if a folder path is provided, assume that the reference solution is there and overwrite the default    
        boundary_cond.overwrite_ref_path(folder)

    net = NeuralNetwork(boundary_cond)
    net.load_checkpoint(folder+f"/checkpoints/net_checkpoint_{epoch}.pt")

    plotter = PlotFuns(boundary_cond)
    plotter.setresultsdir(folder)

    return net, plotter 

def checkpoint_plots_all(logger,boundary_cond,net,plotter):

    folder = logger.get_results_path()
    compute_sde(net,boundary_cond.Tf,4000,plotter,boundary_cond)
    plotter.plot_drift(times = [0,boundary_cond.Tf/2,boundary_cond.Tf],net=net)


if __name__=="__main__":
    from ..utils.parser import checkpoint_plot_params
    import src.sde.boundary as boundaries

    params = checkpoint_plot_params()
    boundary_cond = getattr(boundaries,params.boundary)()
    
    net, plotter = load_net(boundary_cond,params.epoch,params.folder)      
    compute_sde(net,boundary_cond.Tf,params.samples,plotter,boundary_cond)
    plotter.plot_drift(times = [0,boundary_cond.Tf/2,boundary_cond.Tf],net=net)
