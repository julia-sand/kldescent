#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1

module load pytorch
python src --device="cuda" --tsteps=512 --logstep=1 --batchsamples=64 --epochs=5000 --terminalcost="GirsanovMC" --duhiddenlayerneurons=64 --folder="../../../../scratch/project_2011332/kldescent_nonlinear_test_gpu1" --weight=5 --boundary="UnderdampedLandauer" --testsamples=65536 

