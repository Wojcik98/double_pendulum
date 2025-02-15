import math
import os
import random
from collections import deque, namedtuple
from typing import Callable, List
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from q_network import QNetwork
from rsl_rl.env import VecEnv
from rsl_rl.modules.utils import get_activation

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN:
    def __init__(
        self,
        env: VecEnv,
        device: torch.device,
        batch_size: int = 128,
        gamma: float = 0.99,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 200,
        tau: float = 5e-3,
        learning_rate: float = 1e-4,
        activation: str = "elu",
        hidden_dims: List[int] = [64, 64],
        mem_size: int = 10000,
        policy_sampling: int = 10,
        train_cbs: List[Callable[["DQN", dict], None]] = [],
        eval_cbs: List[Callable[["DQN", dict], None]] = [],
        **kwargs,
    ) -> None:
        self.env = env
        self.device = device
        self.num_envs = env.num_envs

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.policy_sampling = policy_sampling

        self.num_actions = env.num_actions
        self.num_obs = env.num_obs

        self.policy_net = QNetwork(
            state_size=self.num_obs,
            action_size=self.num_actions,
            activation=activation,
            hidden_dims=hidden_dims,
        ).to(device)
        self.target_net = QNetwork(
            state_size=self.num_obs,
            action_size=self.num_actions,
            activation=activation,
            hidden_dims=hidden_dims,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(capacity=mem_size)

        self.steps_done = 0
        self._current_episode_lengths = torch.zeros(self.num_envs, dtype=torch.float)
        self._current_cumulative_rewards = torch.zeros(self.num_envs, dtype=torch.float)

        self.train_cbs = train_cbs
        self.eval_cbs = eval_cbs

    def eval(self, num_steps: int, return_epochs: int = 100) -> None:
        self.eval_mode()
        self._episode_statistics = {
            "lengths": [],
            "returns": [],
        }

        state = self.env.reset()

        self._current_episode_lengths = torch.zeros(self.num_envs, dtype=torch.float)
        self._current_cumulative_rewards = torch.zeros(self.num_envs, dtype=torch.float)

        for i in range(num_steps):
            action = self.policy_net.get_best_action(
                state, num_samples=self.policy_sampling
            )
            next_state, reward, done, infos = self.env.step(action)

            self._current_episode_lengths += 1
            self._current_cumulative_rewards += reward.cpu()

            completed_lengths = self._current_episode_lengths[done.cpu().bool()].cpu()
            completed_returns = self._current_cumulative_rewards[
                done.cpu().bool()
            ].cpu()
            self._episode_statistics["lengths"].extend(completed_lengths.tolist())
            self._episode_statistics["returns"].extend(completed_returns.tolist())
            self._episode_statistics["lengths"] = self._episode_statistics["lengths"][
                -return_epochs:
            ]
            self._episode_statistics["returns"] = self._episode_statistics["returns"][
                -return_epochs:
            ]
            self._current_episode_lengths[done.cpu().bool()] = 0.0
            self._current_cumulative_rewards[done.cpu().bool()] = 0.0

            state = next_state

            for cb in self.eval_cbs:
                cb(self, self._episode_statistics)

    def train(
        self,
        num_iterations: int,
        num_steps_per_env: int,
        return_epochs: int = 100,
        num_batches: int = 5,
    ) -> None:
        self.train_mode()
        self._episode_statistics = {
            "lengths": [],
            "returns": [],
            "loss": {},
            "current_iteration": 0,
            "collection_time": 0,
            "total_time": 0,
            "training_time": 0,
            "update_time": 0,
        }

        state = self.env.reset()

        self._current_episode_lengths = torch.zeros(self.num_envs, dtype=torch.float)
        self._current_cumulative_rewards = torch.zeros(self.num_envs, dtype=torch.float)

        for i_episode in range(num_iterations):
            self._episode_statistics["current_iteration"] = i_episode
            start = time.time()

            # Collect data
            for t in range(num_steps_per_env):
                # print(f"state shape: {state.shape}")
                action = self.select_action(state)
                # print(f"action shape: {action.shape}")
                next_state, reward, done, infos = self.env.step(action)
                dones_idx = done.cpu().bool()

                self.memory.push(state, action, next_state, reward, done)

                state = next_state

            self._episode_statistics["collection_time"] = time.time() - start
            start = time.time()

            # Optimize model
            for _ in range(num_batches):
                self.optimize_model()

            self._episode_statistics["update_time"] = time.time() - start

            # Stats

            self._episode_statistics["total_time"] = (
                self._episode_statistics["collection_time"]
                + self._episode_statistics["update_time"]
            )
            self._episode_statistics["training_time"] += self._episode_statistics[
                "total_time"
            ]

            self._current_episode_lengths += 1
            self._current_cumulative_rewards += reward.cpu()

            completed_lengths = self._current_episode_lengths[dones_idx].cpu()
            completed_returns = self._current_cumulative_rewards[dones_idx].cpu()
            self._episode_statistics["lengths"].extend(completed_lengths.tolist())
            self._episode_statistics["returns"].extend(completed_returns.tolist())
            self._episode_statistics["lengths"] = self._episode_statistics["lengths"][
                -return_epochs:
            ]
            self._episode_statistics["returns"] = self._episode_statistics["returns"][
                -return_epochs:
            ]
            self._current_episode_lengths[dones_idx] = 0.0
            self._current_cumulative_rewards[dones_idx] = 0.0

            for cb in self.train_cbs:
                cb(self, self._episode_statistics)

    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)
        non_final_mask = ~(done_batch.bool())
        non_final_next_states = state_batch[non_final_mask]

        state_action_batch = torch.cat((state_batch, action_batch), 2)
        state_action_values = self.policy_net(state_action_batch)

        next_state_values = torch.zeros(
            self.batch_size, self.num_envs, device=self.device
        )
        next_state_values[non_final_mask] = self.target_net.get_best_q_value(
            non_final_next_states, num_samples=self.policy_sampling
        )

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(-1)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self._episode_statistics["loss"] = {"Q function": loss.item()}

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = (
                self.tau * policy_net_state_dict[key]
                + (1 - self.tau) * target_net_state_dict[key]
            )
        self.target_net.load_state_dict(target_net_state_dict)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * (
            math.exp(-1.0 * self.steps_done / self.eps_decay)
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net.get_best_action(
                    state, num_samples=self.policy_sampling
                )
        else:
            return torch.rand((*state.shape[:-1], 1), device=self.device) * 2 - 1

    def save(self, path: str) -> None:
        content = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(content, path)

    def load(self, path: str) -> None:
        content = torch.load(path)
        self.policy_net.load_state_dict(content["policy_net"])
        self.target_net.load_state_dict(content["target_net"])
        self.optimizer.load_state_dict(content["optimizer"])

    def get_inference_policy(
        self,
        num_samples: int = 10,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Returns a policy that chooses the best action"""
        self.eval_mode()

        def policy(state: torch.Tensor) -> torch.Tensor:
            return self.policy_net.get_best_action(state, num_samples=num_samples)

        return policy

    def eval_mode(self) -> None:
        self.policy_net.eval()
        self.target_net.eval()

    def train_mode(self) -> None:
        self.policy_net.train()
        self.target_net.train()

    def _log(self, stat, prefix=""):
        """Logs the progress and statistics of the runner."""
        current_iteration = stat["current_iteration"]

        collection_time = stat["collection_time"]
        update_time = stat["update_time"]
        total_time = stat["total_time"]
        collection_percentage = 100 * collection_time / total_time
        update_percentage = 100 * update_time / total_time

        print("\n" + "=" * 80)
        print(f"iteration:\t{current_iteration}")
        print(
            f"iteration time:\t{total_time:.4f}s (collection: {collection_time:.2f}s [{collection_percentage:.1f}%], update: {update_time:.2f}s [{update_percentage:.1f}%])"
        )

        mean_reward = (
            sum(stat["returns"]) / len(stat["returns"])
            if len(stat["returns"]) > 0
            else 0.0
        )
        mean_steps = (
            sum(stat["lengths"]) / len(stat["lengths"])
            if len(stat["lengths"]) > 0
            else 0.0
        )
        # total_steps = current_iteration * self.env.num_envs * self._num_steps_per_env
        # sample_count = stat["sample_count"]
        print(f"avg. reward:\t{mean_reward:.4f}")
        print(f"avg. steps:\t{mean_steps:.4f}")
        # print(f"stored samples:\t{sample_count}")
        # print(f"total steps:\t{total_steps}")

        for key, value in stat["loss"].items():
            print(f"{key} loss:\t{value:.4f}")

        # for key, value in self.agent._bm_report().items():
        #     mean, count = value
        #     print(
        #         f"BM {key}:\t{mean/1000000.0:.4f}ms ({count} calls, total {mean*count/1000000.0:.4f}ms)"
        #     )

        # self.agent._bm_flush()
