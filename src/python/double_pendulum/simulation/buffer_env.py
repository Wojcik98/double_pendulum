from enum import Enum, auto
from typing import Callable, Optional

import numpy as np
import torch
from double_pendulum.model.torch_plant import Integrator, PlantParams
from double_pendulum.simulation.torch_env import ResetFun, RewardFun, Robot, TorchEnv
from initial_buffer.algorithms.projection_buffer import ProjectionBuffer


class BufferEnv(TorchEnv):
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        max_episode_length: int,
        plant_params: PlantParams,
        integrator: Integrator = Integrator.RUNGE_KUTTA,
        reset_fun: Optional[ResetFun] = None,
        reward_fun: Optional[RewardFun] = None,
        termination_reward: float = -1.0,
        robot: Robot = Robot.ACROBOT,
        gamma: float = 0.99,
    ):
        super().__init__(
            num_envs,
            device,
            max_episode_length,
            plant_params,
            integrator,
            reset_fun,
            reward_fun,
            termination_reward,
            robot,
        )

        self.projection_buffer = ProjectionBuffer(
            device=device,
            nr_clusters=10,
            cluster_algo="kmeans",
            obs_dim=4,
            advantage_gamma=gamma,
            gae_lambda=0.95,
            sampling_strategy="network",
        )

    def reset(self, idxs: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict]:
        if idxs is None:
            idxs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if self._reset_fun is None:
            new_state = torch.zeros((self.num_envs, 4), device=self.device)
        else:
            new_state = self._reset_fun(self.num_envs, self.device)

        new_state = self._normalize_state(new_state)
        self.state[idxs] = new_state[idxs]
        self.state = self._trim_state(self.state)
        self.infos["observations"]["policy"] = self.state
        return self.get_observations()[0]  # return only the state

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        torques = self._denormalize_action(actions)
        # torques = actions
        if self.robot == Robot.ACROBOT:
            torques = torch.cat((self._zeros, torques), dim=1)
            # torques = torch.cat((self._zeros, 10 * torques), dim=1)
        else:
            torques = torch.cat((torques, self._zeros), dim=1)
            # torques = torch.cat((10 * torques, self._zeros), dim=1)

        true_state = self._denormalize_state(self.state)
        # true_state = self.state
        next_state = self._dynamics(true_state, torques)

        self.state = self._normalize_state(next_state)
        # self.state = next_state
        true_state = next_state
        # print(true_state[:, :2].min(), true_state[:, :2].max())
        self.infos = {"observations": {"policy": self.state}}

        self.step_counter += 1
        timeouts = self.step_counter >= self.max_episode_length
        terminations = self._get_terminations(true_state)
        dones = timeouts | terminations
        self.infos["time_outs"] = timeouts.to(dtype=torch.long)
        self.step_counter[dones] = 0
        self.reset(dones)
        dones = dones.to(dtype=torch.long)

        rewards = self._get_rewards(true_state, torques, self.prev_action)
        rewards[terminations] += self._termination_reward
        self.prev_action = torques
        return self.state, rewards, dones, self.infos
