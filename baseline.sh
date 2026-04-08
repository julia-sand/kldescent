#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

module load pytorch
cd src
python -m kldescent.benchmark.benchmarksol --duhiddenlayerneurons=64 --folder="../../../../../scratch/project_2011332/BASELINE_NONLIN_3" --weight=20 --boundary="UnderdampedLandauer" --cellxnum=100000 --celltnum=128 --usebenchmark=True
