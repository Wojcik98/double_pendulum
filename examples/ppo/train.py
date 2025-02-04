import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.torch_plant import Integrator, PlantParams, TorchPlant
from double_pendulum.simulation.torch_env import Robot, TorchEnv
from rsl_rl.runners import OnPolicyRunner
from utils import load_plant_params, parse_args


def main():
    aio_cfg, rl_cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    num_envs = aio_cfg.num_envs

    # simulation parameter
    dt = aio_cfg.dt
    max_episode_length = aio_cfg.max_episode_length_steps
    integrator = Integrator.RUNGE_KUTTA
    # start = [0.0, 0.0, 0.0, 0.0]
    # goal = [np.pi, 0.0, 0.0, 0.0]
    plant_params = load_plant_params(num_envs, device)
    plant = TorchPlant(plant_params)

    def reset_fun(num_envs: int, device: torch.device) -> torch.Tensor:
        mean = torch.zeros(num_envs, 4, device=device)
        mean[:, 0] = torch.pi
        std = 1.5
        return torch.normal(mean, std)

    def reward_fun(
        state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        theta = state[:, :2]
        kin = plant.forward_kinematics(theta)
        ee_x = kin[:, 1, 0]
        ee_y = kin[:, 1, 1]
        target_y = 0.5

        y_diff = target_y - ee_y
        # height_reward = torch.exp(torch.abs(ee_y)) * torch.sign(ee_y)
        height_reward = torch.exp(ee_y)
        # height_reward = torch.exp(-y_diff)
        x_reward = torch.exp(-ee_x.abs())

        # vel = (state[:, 2:] ** 2).mean(dim=1)
        vel = state[:, 2:].abs().sum(dim=1)

        torques = action
        torques_penalty = (torques**2).mean(dim=1)

        action_diff = prev_action - action
        action_diff_penalty = (action_diff**2).mean(dim=1)

        target = torch.tensor([[torch.pi, 0.0]], device=device)
        diff_angle = torch.abs(theta - target)
        diff_term = torch.exp(-diff_angle)  # reward for being close to target
        # add reward for elbow only if shoulder close to the target
        # diff_term[:, 1] *= diff_angle[:, 0] < np.deg2rad(5.0)
        diff_term[:, 1] *= diff_angle[:, 0] ** 2
        diff_term = diff_term.sum(dim=1)

        x_diff_penalty = ee_x.abs()

        reward = (
            height_reward
            - 5e-2 * vel
            - 4e-2 * torques_penalty
            - 1e-2 * action_diff_penalty
            # - 4e-2 * x_diff_penalty
            + 1e-1 * x_reward
            + 1e-1 * diff_term
        )

        return reward

    env = TorchEnv(
        num_envs,
        device,
        max_episode_length=max_episode_length,
        plant_params=plant_params,
        integrator=integrator,
        reset_fun=reset_fun,
        reward_fun=reward_fun,
        termination_reward=0.0,
    )

    log_root_path = os.path.join("logs", rl_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if rl_cfg.run_name:
        log_dir += f"_{rl_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    runner = OnPolicyRunner(env, asdict(rl_cfg), log_dir=log_dir, device=device)

    runner.learn(num_learning_iterations=rl_cfg.max_iterations)


if __name__ == "__main__":
    main()
