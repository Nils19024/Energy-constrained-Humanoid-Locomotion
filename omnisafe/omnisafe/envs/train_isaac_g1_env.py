# FILE: /work/dlclarge2/schmidtn-schmidt-IsaacSim/New/omnisafe/omnisafe/envs/isaac_g1_env.py
# (Corrected Version)

from omnisafe.envs.core import CMDP, env_register
import numpy as np
import torch
import gymnasium
from gymnasium.spaces import Box

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

@env_register
class TrainIsaacFlatEnv(CMDP):
    _support_envs = ["Isaac-Velocity-Flat-G1-v0-Train"]
    need_time_limit_wrapper = False 
    need_auto_reset_wrapper = False
    need_evaluation = False 

    def __init__(self, env_id: str, num_envs: int = 1, device: str = "cpu", **kwargs):
        super().__init__(env_id)
        
        self._num_envs = num_envs
        self._device = torch.device(device)
        
        env_cfg = G1FlatEnvCfg()
        env_cfg.scene.num_envs = self._num_envs
        env_cfg.sim.device = str(self._device)
        
        self._env = ManagerBasedRLEnv(cfg=env_cfg)
        
        obs_shape = self._env.observation_manager.group_obs_dim["policy"]
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        
        isaac_act_space = self._env.action_space
        action_dim = isaac_act_space.shape[-1]

        self._action_space = Box(
            low=np.full((action_dim,), -1.0, dtype=np.float32),
            high=np.full((action_dim,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        print("Observation space:", self._observation_space, "Action space:", self._action_space)

        self._cost_space = Box(shape=(1,), low=-np.inf, high=np.inf, dtype=np.float32)
        self.default_joint_pos = self._env.unwrapped.scene["robot"].data.default_joint_pos.clone()

    @property
    def max_episode_steps(self) -> int:
        return self._env.max_episode_length

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        obs = obs_dict["policy"]
        print("Observation:", obs)
        cost = torch.zeros_like(terminated, dtype=torch.float32)
        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        obs_dict, info = self._env.reset(seed=seed, options=options)
        return obs_dict["policy"], info

    def set_seed(self, seed: int):
        self.reset(seed=seed)

    def render(self):
        pass

    def close(self):
        self._env.close()