#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

module load pytorch
source klenv/bin/activate
python . --tsteps=16 --epochs=1000 --terminalcost="MLE" --duhiddenlayerneurons=16 --folder="../../../../scratch/project_2011332/kldescent1/GAUSS" --weight=1 --logstep=10 --boundary="UnderdampedGaussian" --sigmafinal=4
