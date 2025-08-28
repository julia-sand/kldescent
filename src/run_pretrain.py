import os

import torch

from src.model.train import Pretrain

def run_pretrain(net, 
                path_to_pretrain,
                boundary,
                ):
    
    #fetch pretraining path from command line (or use default)
    #path should be given from cwd
    path_to_pretrain = boundary.get_ref_path()

    #make a folder for pretraining results
    os.makedirs(newpath+"/pretraining", exist_ok = True) #add checkpoints folder

    #save checkpoint so we have a record of the intialised params (or resaved checkpoint)
    torch.save(net.state_dict(),newpath+"/pretraining/initialisation.pt") 

    print("Starting pre-training")
    
    pretrainer = train.Pretrain(path_to_pretrain)

    try:
        train.Pretrain(path_to_pretrain).pretraining_perturb(net)
        print("Pretraining completed successfully.")

    # save if job is manually interrupted
    except KeyboardInterrupt:
        torch.save(net.state_dict(),newpath+"/pretraining/interrupted.pt") 
        print("Pretraining interrupted. Current parameters saved.")

if __name__=="__main__":
    run_pretrain()