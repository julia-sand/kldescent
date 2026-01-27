import torch
import numpy as np 

from ..refsolution.reference import RefSolution
from ..model.nn import save_nn_checkpoint

class Metrics():
    """
    Class for finding metrics during the training process 

    Method: 
        update_metrics
    """

    def __init__(self, net, integrator, logger, refsol=None):
        """
        Initialises class parameters
        args:
            net : class

            integrator : 

            logger : 
            use_perturbative_metrics : Bool. 
                Default False. 
                Set to true to use the perturbative solution as the 
        """
        self.integrator = integrator
        self.net = net

        self.logger = logger
        
        if refsol is not None:

            self.reference_sol = refsol #RefSolution(integrator.boundary,folder_name=self.logger.newpath)
            self.refnet = self.reference_sol.duref_net 
        else: 
            pass
            #self.refnet = None 

        self.update_metrics(0, 0)

    def update_metrics(self, epoch, loss_out, trainer=None):
        """
        Function for computing metrics
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
        
            with torch.no_grad():
                
                #this also updates the ref. cumulants
                kl_dynamic, kl_static, mean0, var0 = self.integrator.compute_metrics(self.refnet)

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
            