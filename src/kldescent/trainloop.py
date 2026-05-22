from kldescent.model.nn import save_nn_checkpoint

def train_loop_du(epoch,
                    metric_,
                    terminal_cost,
                    integrator,
                    trainer,
                    logstep):
    """
    Run a single training epoch.

    Parameters
    ----------
    epoch : int
        Current training epoch index.

    metric_ : object
        Metric evaluator used to compute training diagnostics.

    terminal_cost : object
        Object defining the terminal cost evaluation method used in the optimisation objective.

    integrator : object
        Object defining the Langevin dynamics and integration method used to propagate dynamics.

    trainer : object
        Trainer object responsible for updating model parameters.

    logstep : int
        Frequency (in iterations or epochs) at which metrics are logged.

    Returns
    -------
    None
    """
    
    #update the terminal cost with current final distribution
    terminal_cost.update_cost_params(integrator,epoch)

    #for _ in range(batches):
    #update the drift nn with the current learned final distribution  
    loss_out = trainer.train_step()
        
    if loss_out!=loss_out:
        raise ValueError('Loss function is NaN. Terminating training process')

    #if epoch<trainer.milestone:
    #trainer.scheduler1.step()#metric_.actioncost)
    
    #    metric_.lr = trainer.scheduler1.get_last_lr()[0]
    #else:
    
    
    trainer.scheduler1.step()
    #trainer.scheduler2.step()

    if epoch>trainer.start_decay:
    
        #metric_.update_metrics(epoch, loss_out)
        trainer.scheduler2.step()

        #trainer.scheduler2.step(metric_.actioncost)
        #trainer.scheduler1.step()

    if epoch % logstep == 0:
        
        #if epoch<trainer.start_decay:
        metric_.update_metrics(epoch, loss_out)

        #save checkpoint 
        save_nn_checkpoint(metric_.net,loc=metric_.metrics_logger.newpath+f"/checkpoints/{metric_.net.net_name}_checkpoint_{epoch}.pt")

        #uncomment to make plots
        #!! creating plots will run perturbative calculations
        #plot_utils = PlotUtils(epoch=epoch,                                  
        #                        benchmark=metric_.benchmark_,
        #                        wandb_run = metric_.wandb_run)

        #plot_utils.checkpoint_plots_du(usebenchmark=metric_.usebenchmark,mcsamples=2048)
        
