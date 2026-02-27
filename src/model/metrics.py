import torch
import numpy as np 

from ..refsolution.reference import RefSolution
from ..model.nn import save_nn_checkpoint
from ..sde.kl import static_kl_divergence

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

        #self.update_metrics(0, 0)
        self.kldyn = 0
        self.lr = 0 


    def update_metrics(self, epoch, loss_out):
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
                
                kl_dynamic,xinit,xref = self.integrator.compute_metrics(self.reference_sol)

                kl_static_from_ref,kl_static = static_kl_divergence(xinit,xref,self.integrator.boundary)
                self.actioncost = kl_dynamic + self.integrator.boundary.params["weight"]*kl_static_from_ref

                self.logger.write_data_row(epoch,
                                            loss_out,
                                            self.actioncost,
                                            #kl_dynamic + self.integrator.boundary.params["weight"]*kl_static_from_ref,
                                            kl_dynamic,
                                            kl_static,
                                            kl_static_from_ref,
                                            self.lr)
                                           # np.nanvar(xinit[...,0]))

                #net_name = self.net.net_name


                #save checkpoint 
                #save_nn_checkpoint(self.net,loc=self.logger.newpath+f"/checkpoints/{self.net.net_name}_checkpoint_{epoch}.pt")
        