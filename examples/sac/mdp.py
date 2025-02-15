import torch
from double_pendulum.model.torch_plant import TorchPlant
from typing import List, Callable


def reset_fun(num_envs: int, device: torch.device) -> torch.Tensor:
    mean = torch.zeros(num_envs, 4, device=device)
    mean[:, 0] = torch.pi - 0.05
    # std = 1.5
    std = 0.1
    return torch.normal(mean, std)


def get_reward_fun(
    weights: List[float],
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.device], torch.Tensor
]:
    (
        vel_weight,
        torque_weight,
        action_diff_weight,
        x_diff_weight,
        x_reward_weight,
        diff_term_weight,
    ) = weights

    def reward_fun(
        plant: TorchPlant,
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
        height_reward = torch.exp(ee_y) * (ee_y > 0.4).float()
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
            height_reward / 10
            - vel_weight * vel
            - torque_weight * torques_penalty
            - action_diff_weight * action_diff_penalty
            - x_diff_weight * x_diff_penalty
            + x_reward_weight * x_reward
            + diff_term_weight * diff_term
        )

        return reward

    return reward_fun


# reward_fun = get_reward_fun([0, 1e-3, 1e-3, 1e-2, 1e-1, 1e-1])
reward_fun = get_reward_fun([1e-5, 1e-5, 1e-5, 0, 0, 0])
