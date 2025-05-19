#TODO add all output files created here
import csv
import os
import time

import numpy as np 
import torch

#import memory_profiler

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

    
    def __init__(self):

        ##make a directory for the results
        dirname = os.getcwd()
        #os.path.dirname(__file__) #fetches current location 
        run_id = time.strftime("%Y%m%d-%H%M%S")

        self.newpath = os.path.join(dirname,f'results/{run_id}')
        os.makedirs(self.newpath, exist_ok = True)
        os.makedirs(self.newpath+"/checkpoints", exist_ok = True) #add checkpoints folder
        print("Output Directory created successfully.")   
       
        #create a log file to store the data output
        self.datafile = self.newpath +"/data.txt" 

        self.filename = self.newpath +"/loss.csv" 
        with open(self.filename, 'w') as f:
                fields=['epoch','Lhat','dynamic_kl','static_kl']
                writer = csv.writer(f)
                writer.writerow(fields)
        print("Log file created successfully.")   

    def write_data_row(self,epoch,Lhat,kl_dyn,klval):

        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, Lhat, kl_dyn, klval])

    def save_checkpoint(self,net,epoch):
        torch.save(net.state_dict(),self.newpath+f"/checkpoints/net_checkpoint_{epoch}.pt") 

    def get_epoch_folder(self,epoch):
        os.makedirs(self.newpath+f"/epoch{epoch}", exist_ok = True)
        return f"/epoch{epoch}"

    ##@profile
    def evaluate_net_output(self,net,plot_times):
        #computes neural network output (forward pass) to use in plotting
        dus = np.zeros((1000,len(plot_times)))
        for t in enumerate(plot_times):
            with torch.no_grad():
                dus[:,t[0],None] = net(torch.column_stack((t[1]*torch.ones(1000,dtype=torch.float32)
                                                    ,torch.linspace(-4,4,1000)))).detach().numpy().reshape((1000,1))
        return dus

    #@profile
    def make_epoch_plots(self,epoch,xout,net,plotter):
        #create epoch folder and set plotter subdirectory
        plotter.setsubdir(self.get_epoch_folder(epoch))
        #make plots
        plotter.marginal_distribution(xout[:,0])
        #plotter.plot_drift(np.linspace(-4,4,1000),
        #                    times = [0,0.5,1],
        #                    dus = self.evaluate_net_output(net,[0,0.5,1]))

    #@profile
    def log_epoch_data(self,net,epoch,lout,kl_dyn,kl_val,xout,plotter):

        self.write_data_row(epoch,lout,kl_dyn,kl_val) #log

        #self.make_epoch_plots(epoch,xout,net,plotter)

        self.save_checkpoint(net,epoch)

    def get_results_path(self):
        return self.newpath

    def add_param_to_log(self,name,val):
        
        with open(self.datafile, "a") as f:
            print(str(name)+f"= {val}", file=f)

    def log_args(self,args):
        for arg in vars(args): 
            self.add_param_to_log(arg, getattr(args, arg))
        