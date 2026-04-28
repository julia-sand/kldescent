import csv
import os
import time

import numpy as np 
import torch
import yaml

from kldescent.utils.parser import fetch_results_folder_from_cmd

def save_params(params,logfile):
    """
    Saves parameters to a log file.

    Args:
        params (dict): Parameters to output.
        logfile (str): Path to the file where parameters will be saved.

    Returns:
        None
    """

    with open(logfile, 'a+') as outfile:
        yaml.dump(params, outfile)

class LossLogger():
    """
    Handles creation of result folders, logging of losses, plotting, and saving checkpoints.
    
    Attributes:
        newpath (str): Path to the results directory. Either fetched from command-line
            arguments or created in the current working directory with a timestamp.
        datayaml (str): Path to the YAML file used to store training parameters.
        datafile (str): Path to a .txt file used to log loss data.
        filename (str): Path to the CSV file used to record loss values.

    Methods:
        __init__():
            Initializes the results directory structure, including creating subfolders and log files.
        write_data_row():
            Writes a row of data to the CSV file.

        get_epoch_folder():
            Creates (if necessary) and returns the file path for a given epoch folder.
    """
    
    def __init__(self):
        """
        Creates a timestamped results folder, adds a subfolder for checkpoints,
        and initializes a CSV file to record loss values.

        Returns:
            None
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
