#!/bin/bash
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
source /work/dlclarge2/weisbarj-dllab/conda/bin/activate
conda activate env_isaaclab
module load cuda/12.1
srun ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --headless --video --video_length 400 --num_envs 1 --task Isaac-Velocity-Flat-G1-v0 --checkpoint /work/dlclarge2/weisbarj-dllab/IsaacLab/logs/skrl/g1_flat/2025-08-04_11-05-52_ppo_torch/checkpoints/best_agent.pt