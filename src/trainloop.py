from src.viz.checkpoint_plots import PlotUtils

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
                    
        print(trainer.scheduler.get_last_lr())

        #update metrics 
        metric_.update_metrics(epoch, loss_out)
        
        trainer.scheduler.step(metric_.actioncost)

        if epoch % logstep == 0:
            
            #uncomment to make plots
            #!! creating plots will run perturbative calculations
            plot_utils = PlotUtils(refsol=refsolution_,epoch=epoch+1)

            plot_utils.checkpoint_plots_du()
            #plot_utils.make_girsanov_net_plots()
            #plot_utils.make_contour_plot()
