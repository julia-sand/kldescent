#!/bin/bash
#SBATCH --account=project_2011332
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

module load pytorch
cd src
python -m kldescent.benchmark.benchmarksol --duhiddenlayerneurons=64 --folder="../../../../../scratch/project_2011332/BASELINE_NONLIN" --weight=20 --boundary="UnderdampedLandauer" --cellxnum=50000 --celltnum=1001 --usebenchmark=True
