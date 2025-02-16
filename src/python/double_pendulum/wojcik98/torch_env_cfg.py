from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from double_pendulum.wojcik98.torch_env import TorchEnv


RewardFunc = Callable[[TorchEnv, dict[str, Any]], torch.Tensor]
"""Type alias for reward functions.

Args:
    env: Environment.
    params: Parameters.

Returns:
    Reward, shape: (num_envs,).
"""

TerminationFunc = Callable[[TorchEnv, dict[str, Any]], torch.Tensor]
"""Type alias for termination functions.

Args:
    env: Environment.
    params: Parameters.

Returns:
    Termination signal, shape: (num_envs,).
"""

EventFunc = Callable[[TorchEnv, torch.Tensor, dict[str, Any]], None]
"""Type alias for event functions.

Args:
    env: Environment.
    env_idxs: Indices of the environments.
    params: Parameters.
"""


@dataclass
class RewardTermCfg:
    func: RewardFunc
    weight: float
    params: dict[str, Any] = {}


@dataclass
class RewardsCfg:
    cfg: dict[str, RewardTermCfg]


@dataclass
class TerminationTermCfg:
    func: TerminationFunc
    timeout: bool = False
    """Whether the termination is an episode timeout."""
    params: dict[str, Any] = {}


@dataclass
class TerminationsCfg:
    cfg: dict[str, TerminationTermCfg]


@dataclass
class EventTermCfg:
    class EventMode(Enum):
        STARTUP = auto()
        INTERVAL = auto()
        RESET = auto()

    func: EventFunc
    mode: EventMode
    params: dict[str, Any] = {}


@dataclass
class EventsCfg:
    cfg: dict[str, EventTermCfg]


@dataclass
class DefaultEventsCfg(EventsCfg):
    cfg: dict[str, EventTermCfg] = {}


@dataclass
class TorchEnvCfg:
    num_envs: int
    device: torch.device

    physics_dt: float
    decimation: int

    rewards: RewardsCfg
    terminations: TerminationsCfg
    events: EventsCfg = DefaultEventsCfg()

    @property
    def env_dt(self) -> float:
        return self.physics_dt * self.decimation
