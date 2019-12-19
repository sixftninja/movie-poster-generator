#!/bin/bash

#SBATCH --job-name=cvProject
#SBATCH --nodes=1
#SBATCH --mem=80000
#SBATCH --gres=gpu:p40:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=Output.%j
#SBATCH --mail-user=ama1128@nyu.edu

module load cuda/9.2.88
module load anaconda3/5.3.1
source activate matrix

cd /scratch/ama1128/cvproject
python cvProject.py --epochs 50
