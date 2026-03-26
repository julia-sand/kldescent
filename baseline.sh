#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

module load pytorch
python src --terminalcost="MLE" --duhiddenlayerneurons=16 --folder="../../../../scratch/project_2011332/BASELINE_16" --weight=20 --boundary="UnderdampedLandauer" --testsamples=4096 --usebenchmark=True
