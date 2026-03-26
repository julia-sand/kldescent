#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

module load pytorch
python src --tsteps=16 --terminalcost="MLE" --duhiddenlayerneurons=64 --folder="../../../../scratch/project_2011332/BASELINE_TEST" --weight=20 --boundary="UnderdampedLandauer" --testsamples=4096 --usebenchmark=True
