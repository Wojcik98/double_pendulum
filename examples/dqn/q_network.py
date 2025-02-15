from typing import List

import torch
import torch.nn as nn
from rsl_rl.modules.utils import get_activation


class QNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        activation: str = "elu",
        hidden_dims: List[int] = [64, 64],
    ):
        super().__init__()
        dims = [state_size + action_size] + hidden_dims + [1]
        activation = get_activation(activation)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation)

        self.model = nn.Sequential(*layers)

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.model(state_action)

    def predict(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.model(state_action).detach()

    def get_actions_distribution(self, num_samples: int) -> torch.Tensor:
        assert num_samples % 2 == 1, "num_samples must be odd"
        half = (num_samples // 2) + 1
        x = torch.linspace(0, 2, half)
        y = torch.exp(x)
        y_normalized = (y - y.min()) / (y.max() - y.min())

        actions = torch.cat([y_normalized, -y_normalized[1:]])
        return actions

    def sample_actions(self, state: torch.Tensor, num_samples: int) -> torch.Tensor:
        device = state.device
        if len(state.shape) == 2:
            num_batches = 1
            num_envs = state.shape[0]
            state = state.unsqueeze(0)
            batched = False
        else:
            num_batches = state.shape[0]
            num_envs = state.shape[1]
            batched = True

        actions = self.get_actions_distribution(num_samples).to(device)
        actions = actions.unsqueeze(0).expand(num_batches, num_envs, -1).unsqueeze(-1)

        state = state.unsqueeze(-2)
        state = state.expand(num_batches, num_envs, num_samples, -1)

        state_action = torch.cat([state, actions], dim=-1)
        q_values = self.predict(state_action).squeeze(-1)

        if not batched:
            q_values = q_values.squeeze(0)
        return q_values

    def get_best_action(
        self,
        state: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        q_values = self.sample_actions(state, num_samples)
        best_action_idx = torch.argmax(q_values, dim=-1)
        actions = self.get_actions_distribution(num_samples).to(state.device)
        actions = actions.unsqueeze(0).expand(state.shape[0], -1)
        best_action = actions.gather(1, best_action_idx.unsqueeze(-1))
        return best_action

    def get_best_q_value(
        self,
        state: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        q_values = self.sample_actions(state, num_samples)
        best_q_value = torch.max(q_values, dim=-1).values
        # print(f"Best q value shape: {best_q_value.shape}")
        return best_q_value
