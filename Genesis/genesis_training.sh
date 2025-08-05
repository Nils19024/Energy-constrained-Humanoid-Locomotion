#!/bin/bash
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --time 23:59:59
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
ACTIVATE VENV
srun python3 ./train.py --max_iterations 50 --num_envs 64