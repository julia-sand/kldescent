#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:15:00

module load pytorch
cd src
python -m kldescent.benchmark.benchmarksol --folder="../../../../../scratch/project_2011332/BASELINE_NEW" --weight=20 --boundary="UnderdampedLandauer" --cellxnum=32000 --celltnum=11 --usebenchmark=True
