#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

module load pytorch
source klenv/bin/activate
python . --tsteps=256 --logstep=2 --epochs=100 --terminalcost="MLE" --duhiddenlayerneurons=16 --folder="../../../../scratch/project_2011332/kldescent1/GAUSS_M1_sigma2_v3" --weight=1 --sigmafinal=2 --boundary="UnderdampedGaussian" --testsamples=65536