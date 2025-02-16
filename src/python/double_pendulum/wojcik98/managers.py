from typing import TYPE_CHECKING

import torch

from double_pendulum.wojcik98.torch_env_cfg import (
    EventsCfg,
    EventTermCfg,
    RewardsCfg,
    TerminationsCfg,
)

if TYPE_CHECKING:
    from double_pendulum.wojcik98.torch_env import TorchEnv


class ManagerBase:
    """Base class for managers."""

    def __init__(self, env: TorchEnv) -> None:
        self.env = env

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self) -> torch.device:
        return self.env.device


class RewardsManager(ManagerBase):
    """Manager for rewards."""

    def __init__(self, rewards_cfg: RewardsCfg, env: TorchEnv) -> None:
        super().__init__(env)
        self.rewards = rewards_cfg

        self._rewards_buf = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float) -> torch.Tensor:
        """Compute rewards."""
        self._rewards_buf[:] = 0.0

        for term in self.rewards.cfg.values():
            val = term.func(self.env, term.params) * term.weight * dt
            self._rewards_buf += val

        return self._rewards_buf


class TerminationsManager(ManagerBase):
    """Manager for terminations."""

    def __init__(self, terminations_cfg: TerminationsCfg, env: TorchEnv) -> None:
        super().__init__(env)
        self.terminations = terminations_cfg

        self._truncated_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._terminated_buf = torch.zeros_like(self._truncated_buf)

    @property
    def dones(self) -> torch.Tensor:
        """The net termination signal. Shape is (num_envs,)."""
        return self._truncated_buf | self._terminated_buf

    @property
    def truncated(self) -> torch.Tensor:
        """The timeout signal (reaching max episode length). Shape: (num_envs,)."""
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        """The terminated signal (reaching a terminal state). Shape: (num_envs,)."""
        return self._terminated_buf

    def compute(self) -> torch.Tensor:
        """Compute terminations."""
        # reset computation
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False

        for term in self.terminations.cfg.values():
            val = term.func(self.env, term.params)
            if term.timeout:
                self._truncated_buf |= val
            else:
                self._terminated_buf |= val

        return self.dones


class EventsManager(ManagerBase):
    """Manager for events."""

    def __init__(self, events_cfg: EventsCfg, env: TorchEnv) -> None:
        super().__init__(env)
        self.events = events_cfg

    def apply_startup(
        self,
        env_idxs: torch.Tensor,
    ) -> None:
        """Apply startup events."""
        for term in self.events.cfg.values():
            if term.mode == EventTermCfg.EventMode.STARTUP:
                term.func(self.env, env_idxs, term.params)

    def apply_interval(
        self,
        env_idxs: torch.Tensor,
        dt: float,
    ) -> None:
        """Apply interval events."""
        pass

    def apply_reset(
        self,
        env_idxs: torch.Tensor,
        global_step: int,
    ) -> None:
        """Apply reset events."""
        pass
