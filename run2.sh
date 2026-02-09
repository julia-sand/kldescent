#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00

module load pytorch
python . --tsteps=16 --epochs=30 --duhiddenlayerneurons=16 --rhohiddenlayerneurons=64 --folder="../../../../scratch/project_2011332/kldescent/test6"
