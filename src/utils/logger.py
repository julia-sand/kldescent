#TODO add all output files created here
import csv
import os
import torch
import time
import numpy as np 

class LossLogger():

    def __init__(self):

        ##make a directory for the results
        dirname = os.path.dirname(__file__) #fetches current location 
        run_id = time.strftime("%Y%m%d-%H%M%S")

        self.newpath = os.path.join(dirname,f'../../results/{run_id}')
        os.makedirs(self.newpath, exist_ok = True)
        os.makedirs(self.newpath+"/checkpoints", exist_ok = True) #add checkpoints folder
        print("Output Directory created successfully.")   
       
        #create a log file to store the data output
        self.filename = self.newpath +"/loss.csv" 
        with open(self.filename, 'w') as f:
                fields=['epoch','Lhat','kl_div_final']
                writer = csv.writer(f)
                writer.writerow(fields)
        print("Log file created successfully.")   

    def write_data_row(self,epoch,Lhat,klval):
        data = [epoch, Lhat, klval]

        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def save_checkpoint(self,net,epoch):
        torch.save(net.state_dict(),self.newpath+f"/checkpoints/net_checkpoint_{epoch}.pt") 

    def get_epoch_folder(self,epoch):
        os.makedirs(self.newpath+f"/epoch{epoch}", exist_ok = True)
        return f"/epoch{epoch}"

    def evaluate_net_output(self,net,net_fixed,plot_times):
        #computes neural network output (forward pass)
        # for selected times
        
        dus = []
        for t in plot_times:
            duplot = net(torch.column_stack((t*torch.ones(1000,dtype=torch.float32)
                                                ,torch.linspace(-4,4,1000)))).detach().cpu().numpy()
            duplot_fixed = net_fixed(torch.column_stack((t*torch.ones(1000,dtype=torch.float32)
                                                ,torch.linspace(-4,4,1000)))).detach().cpu().numpy()
            dus.append([duplot,duplot_fixed])  
        return dus

    def make_epoch_plots(self,plotter,epoch,xout,net,net_fixed):
        #create epoch folder and set plotter subdirectory
        plotter.setsubdir(self.get_epoch_folder(epoch))
        #make plots
        plotter.marginal_distribution(xout.detach().cpu().numpy())
        plotter.plot_drift(np.linspace(-4,4,1000),
                            times = [0,0.5,1],
                            dus = self.evaluate_net_output(net,net_fixed,[0,0.5,1]))


    def log_epoch_data(self,plotter,net,net_fixed,epoch,Lhat,xout):

        #get kl divergence from assigned final
        try:
            kl_val = kl.kl_div(xout.detach().cpu().numpy(),boundary.generate_samples_q(args.batch_samples,mode="final"))
        except:
            kl_val = 0
        
        self.write_data_row(epoch,Lhat.mean().item(),kl_val) #log

        self.make_epoch_plots(plotter,epoch,xout,net,net_fixed)
        self.save_checkpoint(net,epoch)

    def get_results_path(self):
        return self.newpath

    