#!/bin/bash
#SBATCH --partition=dllabdlc_gpu-rtx2080
# #SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
source venv/bin/activate
module load cuda/12.1
srun ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 5000