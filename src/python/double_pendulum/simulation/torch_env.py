from enum import Enum, auto
from typing import Callable, Optional

import torch
from double_pendulum.model.torch_plant import Integrator, PlantParams, TorchPlant
from rsl_rl.env import VecEnv

# args: (num_envs, device), returns: state
ResetFun = Callable[[int, torch.device], torch.Tensor]
RewardFun = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.device], torch.Tensor
]


class Robot(Enum):
    ACROBOT = auto()
    PENDUBOT = auto()


class TorchEnv(VecEnv):
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
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        self.integrator = integrator
        self._reset_fun = reset_fun
        self._reward_fun = reward_fun
        self._termination_reward = termination_reward
        self.robot = robot
        self._zeros = torch.zeros((num_envs, 1), device=device)

        self.num_actions = 1  # shoulder or elbow, depending on the robot
        self.num_obs = 4
        self.num_privileged_obs = 0

        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.dt = plant_params.dt

        self.torque_limit = 6.0
        self.pos_limit = 2 * torch.pi
        self.vel_limit = 15.0

        self.state = torch.zeros((self.num_envs, 4), device=self.device)
        self.prev_action = torch.zeros((self.num_envs, 2), device=self.device)
        self.infos = {"observations": {"policy": self.state}}
        self.reset()

        self.plant = TorchPlant(plant_params)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        # TODO normalization
        return self.state, self.infos

    def reset(self, idxs: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict]:
        if idxs is None:
            idxs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if self._reset_fun is None:
            new_state = torch.zeros((self.num_envs, 4), device=self.device)
        else:
            new_state = self._reset_fun(self.num_envs, self.device)

        self.state[idxs] = new_state[idxs]
        self.state = self._trim_state(self.state)
        self.infos = {"observations": {"policy": self.state}}
        return self.get_observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.robot == Robot.ACROBOT:
            actions = torch.cat((self._zeros, 10 * actions), dim=1)
            # actions = torch.cat((self._zeros, actions), dim=1)
        else:
            actions = torch.cat((10 * actions, self._zeros), dim=1)
            # actions = torch.cat((actions, self._zeros), dim=1)

        next_state = self._dynamics(self.state, actions)
        self.state = next_state
        self.infos = {"observations": {"policy": self.state}}

        self.step_counter += 1
        timeouts = self.step_counter >= self.max_episode_length
        self.infos["time_outs"] = timeouts
        terminations = self._get_terminations(self.state)
        dones = timeouts | terminations
        self.step_counter[dones] = 0
        self.reset(dones)

        rewards = self._get_rewards(self.state, actions, self.prev_action)
        rewards[terminations] += self._termination_reward
        self.prev_action = actions
        return self.state, rewards, dones, self.infos

    def _get_rewards(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self._reward_fun is None:
            rewards = torch.ones((self.num_envs,), device=self.device)
        else:
            rewards = self._reward_fun(state, actions, prev_actions, self.device)

        return rewards

    def _dynamics(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Integration."""
        actions = self._trim_actions(actions)
        # print(actions.min(), actions.max())

        if self.integrator == Integrator.RUNGE_KUTTA:
            dx = self._runge_kutta_integrator(state, actions, self.dt)
        else:
            dx = self._euler_integrator(state, actions, self.dt)

        new_state = self._trim_state(state + self.dt * dx)
        return new_state

    def _runge_kutta_integrator(
        self, x: torch.Tensor, u: torch.Tensor, dt: float
    ) -> torch.Tensor:
        k1 = self.plant.rhs(x, u)
        k2 = self.plant.rhs(x + 0.5 * dt * k1, u)
        k3 = self.plant.rhs(x + 0.5 * dt * k2, u)
        k4 = self.plant.rhs(x + dt * k3, u)

        return (k1 + 2.0 * (k2 + k3) + k4) / 6.0

    def _euler_integrator(
        self, x: torch.Tensor, u: torch.Tensor, dt: float
    ) -> torch.Tensor:
        return self.plant.rhs(x, u)

    def _trim_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.clamp(actions, -self.torque_limit, self.torque_limit)

    def _trim_state(self, state: torch.Tensor) -> torch.Tensor:
        # state[:, :2] = torch.clamp(state[:, :2], -self.pos_limit, self.pos_limit)
        # vel[mask] = 0.0
        state[:, 2:] = torch.clamp(state[:, 2:], -self.vel_limit, self.vel_limit)

        return state

    def _get_terminations(self, state: torch.Tensor) -> torch.Tensor:
        pos = state[:, :2]
        mask = (pos < -self.pos_limit) | (pos > self.pos_limit)
        mask = torch.any(mask, dim=1)
        return mask
