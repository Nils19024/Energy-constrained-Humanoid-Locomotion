#!/bin/bash
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
source ../IsaacEnv/bin/activate
module load cuda/12.1
srun ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
--headless \
--video \
--video_length 400 \
--num_envs 1 \
--task Isaac-Velocity-Flat-G1-v0 \
--checkpoint CHECKPOINT_PATH/agent.pt
