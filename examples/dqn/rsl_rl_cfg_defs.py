from dataclasses import MISSING
from typing import List
from configclass import configclass


@configclass
class DQNCfg:
    batch_size: int = 128
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 200
    tau: float = 5e-3
    learning_rate: float = 1e-4
    activation: str = "elu"
    hidden_dims: List[int] = [64, 64]
    mem_size: int = 10000
    policy_sampling: int = 11
    num_steps_per_env: int = 100
