#TODO add all output files created here
import csv
import os
import torch
import time
import numpy as np 

class LossLogger():
    '''

    Attributes
    ----------
    plotter: 

    Methods
    -------
    __init__()
        #creates a results folder 
        #adds a subfolder for checkpoints
        #initialises a csv to record loss values
    
    write_data_row()
        #writes a row of data to the csv
    
    save_checkpoint() 
        #saves parameters of the neural network into the checkpoints folder  
    
    get_epoch_folder()
        #makes a folder for the epoch if it doesn't exist and returns the filepath
    
    evaluate_net_output()
        #evaluates a forward pass of the neural network and returns values to be plotted
    
    make_epoch_plots()
        #calls plotter to make correct plots at the epoch

    '''

    
    def __init__(self,plotter):

        self.plotter = plotter

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

    def evaluate_net_output(self,net,plot_times):
        #computes neural network output (forward pass) to use in plotting
        net.eval()
        dus = []
        for t in plot_times:
            duplot = net(torch.column_stack((t*torch.ones(1000,dtype=torch.float32)
                                                ,torch.linspace(-4,4,1000)))).detach().cpu().numpy()
            dus.append([duplot])  

        net.train()
        return dus


    def make_epoch_plots(self,epoch,xout,net):
        #create epoch folder and set plotter subdirectory
        self.plotter.setsubdir(self.get_epoch_folder(epoch))
        #make plots
        #self.plotter.marginal_distribution(xout)
        self.plotter.plot_drift(np.linspace(-4,4,1000),
                            times = [0,0.5,1],
                            dus = self.evaluate_net_output(net,[0,0.5,1]))


    def log_epoch_data(self,net,epoch,Lhat,kl_val,xout):

        self.write_data_row(epoch,Lhat.mean().item(),kl_val) #log

        self.make_epoch_plots(epoch,xout,net)

        self.save_checkpoint(net,epoch)


    def save_data(self,Lhat,xout,net,epoch):
        
        #compute kl divergence (TODO)
        #try:
        #    kl_val = kl.kl_div(xout,
        #        boundary.generate_samples_q(xout.shape,mode="final"))
        #except:
        kl_val = 0


        self.log_epoch_data(net,epoch,Lhat,kl_val,xout)

        return 

    def get_results_path(self):
        return self.newpath

    