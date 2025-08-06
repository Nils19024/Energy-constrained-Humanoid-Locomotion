# Omnisafe/SafetyGym:

## Important files overview
```omnisafe/train_velocities.py:``` \
Script to train an OmniSafe agent using the PPOLag algorithm on the SafetyHumanoidVelocity-v0 environment. It sets up custom training configurations, initializes the agent, configures logging, and starts the training process.

```safety-gymnasium/safety_gymnasium/tasks/safe_velocity/safety_humanoid_velocity_v0.py:``` \
Defines a modified Humanoid environment based on Gymnasium’s Mujoco Humanoid. The environment computes a reward based on how close the agent's velocity is to a target velocity and applies a large penalty if the agent falls. It calculates a cost using the control action (cost = self.control_cost(action)), tracks episode velocities and costs, and returns their averages at the end of each episode.

```omnisafe/omnisafe/adapter/onpolicy_adapter.py and omnisafe/omnisafe/envs/safety_gymnasium_env.py:``` \
These files were adapted to enable detailed logging of costs and velocity metrics during training and evaluation.

## Installation
### Create and activate a virtual environment
```python3 -m venv OmnisafeEnv``` \
```source OmnisafeEnv/bin/activate```

### Install dependencies
```cd omnisafe``` \
```pip install -e .```

```cd safety-gymnasium``` \
```pip install -e .```

### Start training
```python3 omnisafe/train_velocities.py```

# IsaacLab/Sim:

## Important files overview
```IsaacLab/scripts/reinforcement_learning/skrl/train.py``` \
Script to train an agent. By default, it uses the PPO algorithm, other options are available.

```IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py``` \
File that defines the reward terms. Use this to add custom reward functions or review what is used.

```IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py``` \
File containing interesting configurations, like choice of reward functions and weights or the velocity command for training and evaluation.

## Installation
### Create and activate a virtual environment

Note: You need Python 3.10 to run IsaacLab! \
```cd IsaacLab``` \
```python3 -m venv IsaacEnv``` \
```source IsaacEnv/bin/activate```

Ensure that the latest pip version is used: \
```pip install --upgrade pip```

### Install dependencies
```pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128``` \
```pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com```

### Start training
```./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 5000``` \
Or use the slurm script isaac_training.sh in the IsaacLab folder.


