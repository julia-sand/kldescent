import torch

def compute_sde(net,
                plottime,
                testing_samples,
                plotter,
                boundary,
                integrator):

    """
    Computes the SDE and returns plots 

    ====
    Parameters
    - net: neural network
    ====

    """
        
    t_vec = integrator.testing_time_grid(plottime)

    #initialise x values at \delta_0 (i.e. x=0)
    xinit = torch.tensor(boundary.generate_samples(testing_samples,"initial"),requires_grad=False,dtype=torch.float32)

    with torch.no_grad():

        #self.generate_time_grid()
        for i in range(0,len(t_vec)-1):
            dt = (t_vec[i+1]-t_vec[i])
            
            dw = integrator.generate_innovation(dt,testing_samples)
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
    from ..sde.boundary import UnderdampedHalfBridge
    from .plots import PlotFuns
    from ..sde.langevin import SDEOverdamped


    new_folder = "tests/20250523-174017"

    boundary_cond = UnderdampedHalfBridge()
    net = NeuralNetwork(64,boundary_cond,
                    checkpointpath=new_folder+"/checkpoints/net_checkpoint_1000.pt")
    integrator = SDEOverdamped(boundary_cond)

    plotter = PlotFuns(boundary_cond)
    plotter.setresultsdir(new_folder)
    compute_sde(net,4.9,10000,plotter,boundary_cond,integrator)
