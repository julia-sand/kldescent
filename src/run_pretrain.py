import torch
from model import train as train
import os

def run_pretrain(net, newpath, boundary):
    #fetch pretraining path from command line (or use default)
    #path_to_pretrain = newpath+"/../../src/refsolution/value.csv" #fetches current location 
    path_to_pretrain = boundary.get_ref_path()

    #make a folder for pretraining results
    os.makedirs(newpath+"/pretraining", exist_ok = True) #add checkpoints folder

    #save checkpoint so we have a record of the intialised params (or resaved checkpoint)
    torch.save(net.state_dict(),newpath+"/pretraining/initialisation.pt") 

    print("Starting pre-training")
    
    try:
        train.Pretrain(path_to_pretrain).pretraining_perturb(net)
        print("Pretraining completed successfully.")

    # save if job is manually interrupted
    except KeyboardInterrupt:
        torch.save(net.state_dict(),newpath+"/pretraining/interrupted.pt") 
        print("Pretraining interrupted. Current parameters saved.")

if __name__=="__main__":
    run_pretrain()