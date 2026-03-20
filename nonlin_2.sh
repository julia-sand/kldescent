#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00

module load pytorch
python . --tsteps=128 --logstep=20 --batchsamples=64 --epochs=5000 --terminalcost="GirsanovMC" --duhiddenlayerneurons=64 --folder="../../../../scratch/project_2011332/kldescent_nonlin_new/NONLIN_1" --weight=5 --boundary="UnderdampedLandauer" --testsamples=65536 --usewandb=True

