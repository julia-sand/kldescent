import csv
import os


#TODO: implement some automatic creation of folder names to organise output better
#filename = r'kldescent/results/run1/loss.csv'

#add headings to loss logger.

def create_file(filename=r'kldescent/results/run1/loss.csv'):
    with open(filename, 'w') as f:
        fields=['epoch','kl_loss','kl_div_final']
        writer = csv.writer(f)
        writer.writerow(fields)
    print("Log file created successfully.")    

def log(data,filename=r'kldescent/results/run1/loss.csv'):

    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)