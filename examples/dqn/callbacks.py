import os
import random
import string
import wandb

from dqn import DQN


def make_save_model_cb(directory):
    def cb(runner: DQN, stat: dict):
        path = os.path.join(directory, "model_{}.pt".format(stat["current_iteration"]))
        runner.save(path)

    return cb


def make_interval_cb(callback, interval):
    def cb(runner: DQN, stat: dict):
        if stat["current_iteration"] % interval != 0:
            return

        callback(runner, stat)

    return cb


def make_first_cb(callback):
    uuid = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    def cb(runner: DQN, stat: dict):
        if hasattr(runner, f"_first_cb_{uuid}"):
            return

        setattr(runner, f"_first_cb_{uuid}", True)
        callback(runner, stat)

    return cb


def make_wandb_cb(run):
    def cb(runner: DQN, stat: dict):
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
        training_time = stat["training_time"]

        run.log(
            {
                "mean_rewards": mean_reward,
                "mean_steps": mean_steps,
                "training_time": training_time,
            }
        )

    return cb


def make_wandb_eval_cb(run):
    def cb(runner: DQN, stat: dict):
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

        run.log(
            {
                "mean_rewards": mean_reward,
                "mean_steps": mean_steps,
            }
        )

    return cb
