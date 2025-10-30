#!/bin/bash
#SBATCH --output=output_v2.txt
#SBATCH --nodelist=parma38
#SBATCH --partition=gpu
#SBATCH --gpus=1

srun julia ./script2.jl