#TODO add all output files created here
import csv
import os
import time

import numpy as np 
import torch
import yaml

from .parser import fetch_results_folder_from_cmd

class LossLogger():
    '''
    This class deals with making results folder and recording losses and plots. also saving checkpoints

    Attributes
    ----------
    

    Methods
    -------
    __init__()
       
    write_data_row()
        #writes a row of data to the csv
    
    get_epoch_folder()
        #makes a folder for the epoch if it doesn't exist and returns the filepath
    
    make_epoch_plots()
        #calls plotter to make correct plots at the epoch
    '''
    
    def __init__(self):
        """Create a timestamped results folder  
        Add a subfolder for checkpoints
        Initialise a csv to record loss values
        """

        #check if there is a results folder to use 
        self.newpath = fetch_results_folder_from_cmd()

        if self.newpath is None:
            ##make a directory for the results in
            # the place the file is run
            dirname = os.getcwd()

            self.newpath = os.path.join(dirname,'results/'+f"{time.strftime("%Y%m%d-%H%M%S")}")
        
        os.makedirs(self.newpath, exist_ok = True)
        os.makedirs(self.newpath+"/checkpoints", exist_ok = True) #add checkpoints folder
        print("Output Directory created successfully.")   
       
        #create a log file to store the data output
        self.datayaml = self.newpath +"/data.yaml" 

        #create a log file to store the data output
        self.datafile = self.newpath +"/data.txt" 

        self.filename = self.newpath +"/loss.csv" 
        with open(self.filename, 'w') as f:
                fields=['epoch','Lhat_objective','cost','dynamic_kl','static_kl','static_kl_from_ref',"position_variance"]
                writer = csv.writer(f)
                writer.writerow(fields)
        print("Log file created successfully.")   

    def write_data_row(self,epoch,Lhat,kl_dyn,kl_static,kl_static_from_ref,refmean,refvar):

        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, Lhat, kl_dyn, kl_static, kl_static_from_ref, refmean, refvar])

    def get_epoch_folder(self,epoch):
        os.makedirs(self.newpath+f"/epoch{epoch}", exist_ok = True)
        return f"/epoch{epoch}"

    def get_results_path(self):
        return self.newpath

    def add_param_to_log(self,name,val):
        
        with open(self.datafile, "a") as f:
            print(str(name)+f"= {val}", file=f)

    def log_dict(self,dictparams):
                
        with open(self.datayaml, 'a+') as outfile:
            yaml.dump(dictparams, outfile)

