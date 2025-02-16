from typing import Any

import gymnasium as gym
import torch
from skrl.envs.wrappers.torch.base import Wrapper

from double_pendulum.wojcik98.torch_env import TorchEnv


class TorchEnvWrapper(Wrapper):
    """SKRL wrapper for TorchEnv."""

    _env: TorchEnv

    def __init__(self, env: TorchEnv) -> None:
        super().__init__(env)

        self._vectorized = True

    @property
    def observation_space(self) -> gym.Space:
        """Observation space of a single environment."""
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space of a single environment."""
        return self._env.action_space

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Perform a step in the environments.

        Args:
            actions: Actions to take in the environments.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, rew, term, trunc, info = self._env.step(actions)
        return obs, rew, term, trunc, info

    def reset(self) -> tuple[torch.Tensor, Any]:
        """Reset the environments.

        Returns:
            Initial observation and info.
        """
        obs, info = self._env.reset()
        return obs, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environments."""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environments."""
        self._env.close()
