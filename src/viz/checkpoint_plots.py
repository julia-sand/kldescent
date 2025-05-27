import math
import torch

def compute_sde(net,
                plottime,
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
        
    t_vec = torch.linspace(0,boundary.Tf,100)
    #boundary.get_reference_timegrid(plottime)

    #initialise x values at \delta_0 (i.e. x=0)
    xinit = torch.tensor(boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32)

    with torch.no_grad():

        #self.generate_time_grid()
        for i in range(0,len(t_vec)-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = math.sqrt(dt)*torch.randn((testing_samples,1))
            du = net.forward(torch.column_stack((t_vec[i]*torch.ones(testing_samples,dtype=torch.float32)
                                                    ,xinit[:,0,None])))

            xinit = boundary.euler_maruyama_step(xinit,du,dt,dw,t_vec[i])

    plotter.make_epoch_plots(xinit.detach().numpy(),
                                net,plot_times=[0,0.5*plottime,plottime])

    #free up the tensors
    del xinit 
    del du

    return 


if __name__=="__main__":
    import os 

    from ..model.nn import NeuralNetwork
    from ..sde.boundary import UnderdampedNoMomentum
    from .plots import PlotFuns


    new_folder = "tests/20250523-174017"

    boundary_cond = UnderdampedNoMomentum()
    net = NeuralNetwork(64,boundary_cond,
                    checkpointpath=new_folder+"/checkpoints/net_checkpoint_2000.pt")

    plotter = PlotFuns(boundary_cond)
    plotter.setresultsdir(new_folder)
    compute_sde(net,5,10000,plotter,boundary_cond)
