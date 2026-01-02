import torch
import numpy as np 

from ..refsolution.reference import RefSolution

def save_nn_checkpoint(self,net,loc):
    
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    
    torch.save(net.state_dict(),loc) 

class Metrics():
    """
    Class for finding metrics during the training process 

    Method: 
        update_metrics
    """

    def __init__(self, net, integrator, logstep, logger):
        
        self.integrator = integrator
        self.net = net

        self.logstep = logstep #how frequently to save results
        self.logger = logger
        
        self.reference_sol = RefSolution(integrator.boundary)

        self.update_metrics(0, 0)

    def update_metrics(self, epoch, loss_out, trainer=None):
        """
        Function for computing metrics
            -logs the metrics at logstep
            - checks for nan loss
            TODO (fix logic)- updates the metrics for final distribution, gaussian reference. 
        args:
            epoch:
                pass current epoch to use in logging loss
            loss_out:
                value of the loss from the metrics function
        """
        
        with torch.no_grad():

            if loss_out is not None:
                loss_out = loss_out
            else:
                loss_out = 0 #TODO: compute the loss independently here

            if np.isnan(loss_out):
                raise ValueError('Loss function is NaN. Terminating training process')
        
            if (epoch % self.logstep == 0): #compute metrics 
                with torch.no_grad():
                    #this also updates the ref. cumulants
                    kl_dynamic, kl_static, mean0, var0 =self.integrator.compute_metrics(self.reference_sol.duref_net)

                    if trainer is not None:
                        lr_rate = trainer.scheduler.get_last_lr()[0]
                    else:
                        lr_rate = 0

                    self.logger.write_data_row(epoch,
                                                loss_out,
                                                kl_dynamic,
                                                kl_static,
                                                lr_rate, #,
                                                mean0,
                                                var0)

                    #save checkpoint 
                    save_nn_checkpoint(self.net,loc=self.logger.newpath+f"/checkpoints/{self.net.net_name}_checkpoint_{epoch}.pt")
                    #if self.rhonet is not None:
                    #    self.logger.save_checkpoint(self.rhonet,epoch)
            else:
                pass
            #else: 
            #    with torch.no_grad():
            #        self.integrator.update_ref_statistics()
                    #pass
