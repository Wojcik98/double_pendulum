from typing import Optional

import gymnasium as gym
import torch

from double_pendulum.wojcik98.managers import (
    EventsManager,
    RewardsManager,
    TerminationsManager,
)
from double_pendulum.wojcik98.torch_env_cfg import TorchEnvCfg


class TorchEnv:
    """Base class for torch-based vectorized environments."""

    def __init__(self, cfg: TorchEnvCfg) -> None:
        self.cfg = cfg
        self.num_envs = self.cfg.num_envs
        self.device = self.cfg.device

        self.reward_manager = RewardsManager(self.cfg.rewards, self)
        self.termination_manager = TerminationsManager(self.cfg.terminations, self)
        self.event_manager = EventsManager(self.cfg.events, self)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Perform a step in the environments.

        Args:
            actions: Actions to take in the environments.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        self.apply_actions(actions)
        obs, info = self.get_observations()
        rewards = self.reward_manager.compute(self.cfg.env_dt)
        self.termination_manager.compute()
        terminated = self.termination_manager.terminated
        truncated = self.termination_manager.truncated

        return obs, rewards, terminated, truncated, info

    def reset(
        self, env_idxs: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environments.

        Returns:
            Initial observation and info.
        """
        if env_idxs is None:
            env_idxs = torch.arange(
                self.num_envs, dtype=torch.int64, device=self.device
            )

        self.reset_idxs(env_idxs)
        obs, info = self.get_observations()
        return obs, info

    @property
    def observation_space(self) -> gym.Space:
        """Observation space of a single environment."""
        raise NotImplementedError

    @property
    def action_space(self) -> gym.Space:
        """Action space of a single environment."""
        raise NotImplementedError

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Get observations."""
        raise NotImplementedError

    def apply_actions(self, actions: torch.Tensor) -> None:
        """Apply actions to the environments."""
        raise NotImplementedError

    def reset_idxs(self, idxs: torch.Tensor) -> None:
        """Reset the environments for specific indices."""
        raise NotImplementedError

    def render(self, *args, **kwargs) -> None:
        """Render the environments."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the environments."""
        raise NotImplementedError
