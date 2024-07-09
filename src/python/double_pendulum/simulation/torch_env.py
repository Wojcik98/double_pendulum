from enum import Enum, auto
from typing import Callable, Optional

import torch
from double_pendulum.model.torch_plant import Integrator, PlantParams, TorchPlant
from rsl_rl.env import VecEnv

# args: (num_envs, device), returns: state
ResetFun = Callable[[int, torch.device], torch.Tensor]
RewardFun = Callable[[torch.device], torch.Tensor]


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
        robot: Robot = Robot.ACROBOT,
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_episode_length = max_episode_length
        self.integrator = integrator
        self._reset_fun = reset_fun
        self.robot = robot
        self._zeros = torch.zeros((num_envs, 1), device=device)

        self.num_actions = 1  # shoulder or elbow, depending on the robot
        self.num_obs = 4
        self.num_privileged_obs = 0

        self.step_counter = 0
        self.dt = plant_params.dt

        self.reset()

        self.plant = TorchPlant(plant_params)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        return self.state, self.infos

    def reset(self) -> tuple[torch.Tensor, dict]:
        if self._reset_fun is None:
            self.state = torch.zeros((self.num_envs, 4), device=self.device)
        else:
            self.state = self._reset_fun(self.num_envs, self.device)
        self.infos = {}
        return self.get_observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.robot == Robot.ACROBOT:
            actions = torch.cat((self._zeros, actions), dim=1)
        else:
            actions = torch.cat((actions, self._zeros), dim=1)

        next_state = self._dynamics(self.state, actions)

        self.infos = {}

        self.step_counter += 1
        if self.step_counter == self.max_episode_length:
            self.step_counter = 0
            dones = torch.ones((self.num_envs,), dtype=torch.long, device=self.device)
            self.infos["time_outs"] = dones
        else:
            dones = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        rewards = torch.ones((self.num_envs,), device=self.device)
        self.state = next_state
        return self.state, rewards, dones, self.infos

    def _dynamics(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Runge-Kutta integration."""
        if self.integrator == Integrator.RUNGE_KUTTA:
            dx = self._runge_kutta_integrator(state, actions, self.dt)
        else:
            dx = self._euler_integrator(state, actions, self.dt)
        return state + self.dt * dx

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
