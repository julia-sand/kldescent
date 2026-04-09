#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1

module load pytorch
python src --device="cuda" --tsteps=256 --logstep=10 --batchsamples=64 --epochs=5000 --terminalcost="GirsanovMC" --duhiddenlayerneurons=32 --folder="../../../../scratch/project_2011332/kl_descent_Nonlin_test_2_newsched" --weight=5 --boundary="UnderdampedLandauer" --testsamples=65536 

