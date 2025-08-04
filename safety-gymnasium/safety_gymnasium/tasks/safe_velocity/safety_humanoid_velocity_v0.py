# safety_gymnasium/tasks/safe_velocity/safety_humanoid_velocity_v0.py

import os
import numpy as np
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv, mass_center
from safety_gymnasium.utils.task_utils import add_velocity_marker, clear_viewer

class SafetyHumanoidVelocityEnv(HumanoidEnv):
    """
    Humanoid environment, modifiziert um Episoden-Statistiken im finalen
    `step`-Aufruf zurückzugeben und eine große negative Belohnung für das Hinfallen zu geben.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._velocity_target = 1.84
        self.model.light(0).castshadow = False
        self.sigma = 0.25

        self._episode_velocities = []
        self._episode_costs = []

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        velocity = np.linalg.norm(xy_velocity)
        
        observation = self._get_obs()
        terminated = self.terminated
        info = {}

        velocity_reward = np.exp(- (velocity - self._velocity_target)**2 / (2 * self.sigma**2))
        fall_penalty = 100.0 * float(terminated)
        reward = velocity_reward - fall_penalty
        cost = self.control_cost(action)

        self._episode_velocities.append(velocity)
        self._episode_costs.append(cost)

        if terminated:
            if self._episode_velocities:
                info['episode_metrics'] = {
                    'avg_velocity': np.mean(self._episode_velocities),
                    'avg_cost': np.mean(self._episode_costs)
                }
            self._episode_velocities.clear()
            self._episode_costs.clear()

        return observation, reward, cost, terminated, False, info

    def reset(self, *args, **kwargs):
        self._episode_velocities.clear()
        self._episode_costs.clear()
        
        return super().reset(*args, **kwargs)