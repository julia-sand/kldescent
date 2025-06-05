import math
import torch

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

    with torch.no_grad():

        #self.generate_time_grid()
        for i in range(0,len(t_vec)-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = math.sqrt(dt)*torch.randn((testing_samples,1))
            du = net.forward(torch.column_stack((t_vec[i]*torch.ones(testing_samples,dtype=torch.float32)
                                                    ,xinit[:,0,None])))

            
            xinit = boundary.euler_maruyama_step(xinit,du,dt,dw,t_vec[i])
            xref = boundary.euler_maruyama_step(xref,boundary_cond.du_ref(xref[:,[0]],t_vec[i]),dt,dw,t_vec[i])

    plotter.make_epoch_plots([xinit.numpy(),xref.numpy()],
                                net,marginal_plot_time,drift_plot_times=[0,boundary.Tf/2,boundary.Tf])

    #free up the tensors
    del xinit 
    del du

    return 


if __name__=="__main__":
    import os 

    from ..model.nn import NeuralNetwork
    from ..sde.boundary import UnderdampedHalfGaussian
    from .plots import PlotFuns

    #TODO update so that this reads the parameters straight from the data.txt file

    new_folder = "tests/20250602-163133"

    boundary_cond = UnderdampedHalfGaussian()
    net = NeuralNetwork(boundary_cond)
    net.load_checkpoint(new_folder+"/checkpoints/net_checkpoint_2000.pt")

    plotter = PlotFuns(boundary_cond)
    plotter.setresultsdir(new_folder)
    compute_sde(net,5,5000,plotter,boundary_cond)
