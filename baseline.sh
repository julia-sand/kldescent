#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:15:00

module load pytorch
cd src
python -m kldescent.benchmark.benchmarksol --duhiddenlayerneurons=64 --folder="../../../../../scratch/project_2011332/BASELINE_NONLIN_8" --weight=20 --boundary="UnderdampedLandauer" --cellxnum=2000 --celltnum=11 --usebenchmark=True
