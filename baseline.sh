#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:15:00

module load pytorch
cd src
python -m kldescent.benchmark.overdamped.computesol --folder="../../../../../scratch/project_2011332/BASELINE_NONLIN_M20_v2" --weight=20 --boundary="UnderdampedLandauer" --cellxnum=16000 --celltnum=11 --usebenchmark=True
