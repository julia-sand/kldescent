#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00

module load pytorch
python . --tsteps=32 --epochs=50 --duhiddenlayerneurons=64 --rhohiddenlayerneurons=64 --folder="../../../../scratch/project_2011332/kldescent_M20/TEST2" --weight=20 --logstep=5
