#TODO add all output files created here
import csv
import os
import time

import numpy as np 
import torch
import yaml

#import memory_profiler

class LossLogger():
    '''
    This class deals with making results folder and recording losses and plots. also saving checkpoints

    Attributes
    ----------
    plotter: 

    Methods
    -------
    __init__()
       
    
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
        """
        creates a timestamped results folder  
        adds a subfolder for checkpoints
        initialises a csv to record loss values
        """

        ##make a directory for the results in  the place the file is run
        dirname = os.getcwd()
        run_id = time.strftime("%Y%m%d-%H%M%S")

        self.newpath = os.path.join(dirname,f'results/{run_id}')
        os.makedirs(self.newpath, exist_ok = True)
        os.makedirs(self.newpath+"/checkpoints", exist_ok = True) #add checkpoints folder
        print("Output Directory created successfully.")   
       
        #create a log file to store the data output
        self.datayaml = self.newpath +"/data.yaml" 

        #create a log file to store the data output
        self.datafile = self.newpath +"/data.txt" 

        self.filename = self.newpath +"/loss.csv" 
        with open(self.filename, 'w') as f:
                fields=['epoch','Lhat','dynamic_kl','static_kl','learning_rate', "ref_mean", "ref_var"]
                writer = csv.writer(f)
                writer.writerow(fields)
        print("Log file created successfully.")   

    def write_data_row(self,epoch,Lhat,kl_dyn,klval,lr,refmean,refvar):

        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, Lhat, kl_dyn, klval, lr, refmean, refvar])

    def save_checkpoint(self,net,epoch):
        
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        
        torch.save(net.state_dict(),self.newpath+f"/checkpoints/{net.net_name}_checkpoint_{epoch}.pt") 

    def get_epoch_folder(self,epoch):
        os.makedirs(self.newpath+f"/epoch{epoch}", exist_ok = True)
        return f"/epoch{epoch}"

    #@profile
    def log_epoch_data(self,net,epoch,lout,kl_dyn,kl_val,lr,mean0,var0):

        self.write_data_row(epoch,lout,kl_dyn,kl_val,lr,mean0,var0) #log

        self.save_checkpoint(net,epoch)

    def get_results_path(self):
        return self.newpath

    def add_param_to_log(self,name,val):
        
        with open(self.datafile, "a") as f:
            print(str(name)+f"= {val}", file=f)

    #def log_args(self,args):
    #    for arg in vars(args): 
    #        self.add_param_to_log(arg, getattr(args, arg))
        
    def log_dict(self,dictparams):
                
        with open(self.datayaml, 'a+') as outfile:
            yaml.dump(dictparams, outfile)

