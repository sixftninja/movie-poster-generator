#!/bin/bash

#SBATCH --job-name=embed
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=Output.%j
#SBATCH --mail-user=ama1128@nyu.edu

module load cuda/9.2.88

cd /scratch/ama1128/cvproject
python cvProject.py --epochs 30
