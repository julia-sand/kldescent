import csv
import os


#TODO: implement some automatic creation of folder names to organise output better
#filename = r'kldescent/results/run1/loss.csv'

class Logger():

    #add headings to loss logger.
    def __init__(self,filename):
        self.filename = r'kldescent/results/run1/loss.csv' if filename==None else filename

        with open(self.filename, 'w') as f:
                fields=['epoch','Lhat','kl_div_final']
                writer = csv.writer(f)
                writer.writerow(fields)
        print("Log file created successfully.")    

    def log(self,epoch,Lhat,klval):
        data = [epoch, Lhat, klval]

        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)