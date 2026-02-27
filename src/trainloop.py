from src.viz.checkpoint_plots import PlotUtils
from src.model.nn import save_nn_checkpoint

def train_loop_du(epoch,
                    metric_,
                    terminal_cost,
                    integrator,
                    trainer,
                    refsolution_,
                    logstep):

        #update the terminal cost with current final distribution
        terminal_cost.update_cost_params(integrator,epoch)

        #update the drift nn with the current learned final distribution  
        loss_out = trainer.train_step()
                    
       
        metric.lr = trainer.scheduler.get_last_lr()[0]

        #update metrics 
        metric_.update_metrics(epoch, loss_out)
        
        trainer.scheduler.step(metric_.actioncost)

        if epoch % logstep == 0:
        
            #save checkpoint 
            save_nn_checkpoint(metric_.net,loc=metric_.logger.newpath+f"/checkpoints/{metric_.net.net_name}_checkpoint_{epoch}.pt")
    
            #uncomment to make plots
            #!! creating plots will run perturbative calculations
            plot_utils = PlotUtils(refsol=refsolution_,epoch=epoch)

            plot_utils.checkpoint_plots_du()
            #plot_utils.make_girsanov_net_plots()
            #plot_utils.make_contour_plot()
