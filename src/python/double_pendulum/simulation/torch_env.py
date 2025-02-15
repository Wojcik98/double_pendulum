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
        super().__init__(5, 5, device, num_envs, max_episode_length)

        self.integrator = integrator
        self._reset_fun = reset_fun
        self._reward_fun = reward_fun
        self._termination_reward = termination_reward
        self.robot = robot

        self.num_actions = 1  # shoulder or elbow, depending on the robot

        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.dt = plant_params.dt

        self.torque_limit = 6.0
        self.pos_limit = 2 * torch.pi
        self.vel_limit = 15.0

        self.state = torch.zeros((self.num_envs, 5), device=self.device)
        self.prev_action = torch.zeros((self.num_envs, 2), device=self.device)
        self.infos = {
            "observations": {"policy": self.state},
            "time_outs": torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            ),
        }
        self.reset()

        self.plant = TorchPlant(plant_params)
        self.cfg = plant_params

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        # TODO normalization
        return self.state, self.infos

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        return self.get_observations()

    def reset(self, idxs: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict]:
        if idxs is None:
            idxs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if self._reset_fun is None:
            new_state = torch.zeros((self.num_envs, 5), device=self.device)
        else:
            new_state = self._reset_fun(self.num_envs, self.device)

        new_state = self._normalize_state(new_state)
        self.state[idxs] = new_state[idxs]
        self.infos["observations"]["policy"] = self.state
        return self.get_observations()[0]  # return only the state

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = self._denormalize_action(actions)

        true_state = self._denormalize_state(self.state)
        next_state = self._dynamics(true_state, actions)

        self.state = self._normalize_state(next_state)
        true_state = next_state
        self.infos = {"observations": {"policy": self.state}}

        self.step_counter += 1
        timeouts = self.step_counter >= self.max_episode_length
        terminations = self._get_terminations(true_state)
        dones = timeouts | terminations
        self.infos["time_outs"] = timeouts.to(dtype=torch.long)
        self.step_counter[dones] = 0
        self.reset(dones)
        dones = dones.to(dtype=torch.long)

        rewards = self._get_rewards(true_state, actions, self.prev_action)
        rewards[terminations] += self._termination_reward
        self.prev_action = actions
        return self.state, rewards, dones, self.infos

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        normalized = state.clone()
        normalized[:, :2] /= self.pos_limit
        normalized[:, 2:4] /= self.vel_limit
        normalized[:, 4] /= self.torque_limit

        return normalized

    def _denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        denormalized = state.clone()
        denormalized[:, :2] *= self.pos_limit
        denormalized[:, 2:4] *= self.vel_limit
        denormalized[:, 4] *= self.torque_limit

        return denormalized

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        normalized = action.clone()
        normalized /= self.torque_limit

        return normalized

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        denormalized = action.clone()
        denormalized *= self.torque_limit

        return denormalized

    def _get_rewards(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self._reward_fun is None:
            rewards = torch.ones((self.num_envs,), device=self.device)
        else:
            rewards = self._reward_fun(
                self.plant, state, actions, prev_actions, self.device
            )

        return rewards

    def _dynamics(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Integration."""
        actions = self._trim_actions(actions)
        if self.robot == Robot.ACROBOT:
            torques = torch.cat(
                (torch.zeros((self.num_envs, 1), device=self.device), actions), dim=1
            )
        else:
            torques = torch.cat(
                (actions, torch.zeros((self.num_envs, 1), device=self.device)), dim=1
            )
        state = state[:, :4]

        if self.integrator == Integrator.RUNGE_KUTTA:
            dx = self._runge_kutta_integrator(state, torques, self.dt)
        else:
            dx = self._euler_integrator(state, torques, self.dt)

        new_state = state + self.dt * dx
        new_state = torch.cat((new_state, actions), dim=1)
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

    def _get_terminations(self, state: torch.Tensor) -> torch.Tensor:
        pos = state[:, :2]
        vel = state[:, 2:4]
        mask = (
            (pos < -self.pos_limit)
            | (pos > self.pos_limit)
            # | (vel < -self.vel_limit)
            # | (vel > self.vel_limit)
        )
        mask = torch.any(mask, dim=1)
        return mask
