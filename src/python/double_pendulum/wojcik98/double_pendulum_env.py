import gymnasium as gym
import numpy as np
import torch

from double_pendulum.wojcik98.torch_env import TorchEnv
from double_pendulum.wojcik98.torch_env_cfg import TorchEnvCfg
from double_pendulum.wojcik98.torch_plant import Integrator, PlantParams, TorchPlant


class DoublePendulumEnvCfg(TorchEnvCfg):
    pass


class DoublePendulumEnv(TorchEnv):
    def __init__(self, cfg: DoublePendulumEnvCfg) -> None:
        super().__init__(cfg)
        self.num_actions = 1  # shoulder or elbow, depending on the robot
        self.num_obs = 5  # [theta, omega] for each joint and previous action

        # TODO: Initialize the plant

        self._obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_obs,), dtype=np.float32
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-2.0, high=2.0, shape=(self.num_actions,), dtype=np.float32
        )

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self._obs_buf, {}

    def apply_actions(self, actions: torch.Tensor) -> None:
        # TODO: apply actions to the plant
        pass

    def reset_idxs(self, idxs: torch.Tensor) -> None:
        # TODO: reset the plant for the given indices
        pass

    def render(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass
