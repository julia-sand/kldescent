#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

module load pytorch
source klenv/bin/activate
python . --tsteps=16 --epochs=100 --terminalcost="MLE" --duhiddenlayerneurons=16 --folder="../../../../scratch/project_2011332/kldescent1/GAUSS" --weight=1 --boundary="UnderdampedGaussian" --testsamples=8192