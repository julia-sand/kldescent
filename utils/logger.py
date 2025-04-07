import csv
import os


#TODO: implement some automatic creation of folder names to organise output better
#filename = r'kldescent/results/run1/loss.csv'

#add headings to loss logger.

class Logger():

    def __init__(self,filename):
        self.filename = r'kldescent/results/run1/loss.csv' if filename==None else filename

        with open(self.filename, 'w') as f:
                fields=['epoch','kl_loss','kl_div_final']
                writer = csv.writer(f)
                writer.writerow(fields)
        print("Log file created successfully.")    

    def log(self,data):

        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)