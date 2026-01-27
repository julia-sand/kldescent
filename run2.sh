#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

module load pytorch
python . --tsteps=16 --boundary="UnderdampedLandauer" --testsamples=8192 --epochs=30 --duhiddenlayerneurons=16 --rhohiddenlayerneurons=16
