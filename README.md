# Omnisafe/SafetyGym training

## Important files overview
```omnisafe/train_velocities.py:``` \
Script to train an OmniSafe agent using the PPOLag algorithm on the SafetyHumanoidVelocity-v0 environment. It sets up custom training configurations, initializes the agent, configures logging, and starts the training process.

```safety-gymnasium/safety_gymnasium/tasks/safe_velocity/safety_humanoid_velocity_v0.py:``` \
Defines a modified Humanoid environment based on Gymnasium’s Mujoco Humanoid. The environment computes a reward based on how close the agent's velocity is to a target velocity and applies a large penalty if the agent falls. It calculates a cost using the control action (cost = self.control_cost(action)), tracks episode velocities and costs, and returns their averages at the end of each episode.

```omnisafe/omnisafe/adapter/onpolicy_adapter.py and omnisafe/omnisafe/envs/safety_gymnasium_env.py:``` \
These files were adapted to enable detailed logging of costs and velocity metrics during training and evaluation.

## Installation
### Create an activate a virtuell environment
```python3 -m venv OmnisafeEnv``` \
```source OmnisafeEnv/bin/activate```

### Install dependencies in this order:
```cd omnisafe``` \
```pip install -e .```

```cd safety-gymnasium``` \
```pip install -e .```

### Start training
```python3 omnisafe/train_velocities.py```

# IsaacLab/Sim training

Instructions here

