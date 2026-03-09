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
                    
        #if epoch<trainer.milestone:
        #trainer.scheduler1.step()#metric_.actioncost)
      
        #    metric_.lr = trainer.scheduler1.get_last_lr()[0]
        #else:
        metric_.update_metrics(epoch, loss_out)
        trainer.scheduler2.step(metric_.actioncost)
        metric_.lr = trainer.scheduler2.get_last_lr()[0]


        if epoch % logstep == 0:
        
            #if epoch<trainer.milestone:

            #update metrics 
            #    metric_.update_metrics(epoch, loss_out)
            #else:
            #    pass

            #save checkpoint 
            save_nn_checkpoint(metric_.net,loc=metric_.logger.newpath+f"/checkpoints/{metric_.net.net_name}_checkpoint_{epoch}.pt")
    
            #uncomment to make plots
            #!! creating plots will run perturbative calculations
            plot_utils = PlotUtils(refsol=refsolution_,epoch=epoch)

            plot_utils.checkpoint_plots_du(mcsamples=2048)
            