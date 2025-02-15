import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import wandb
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.torch_plant import Integrator, PlantParams, TorchPlant
from double_pendulum.simulation.torch_env import Robot, TorchEnv
from mdp import reset_fun, get_reward_fun
from rl_cfg import AIOlympicsRunnerCfg, AIOlympicsPPOCfg
from rsl_rl.algorithms import PPO
from rsl_rl.runners import Runner
from rsl_rl.runners.callbacks import make_interval_cb, make_save_model_cb, make_final_cb
from utils import AIOlympicsCfg, load_plant_params
from wandb_config import WANDB_API_KEY, WANDB_ENTITY

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

sweep_config = {
    "method": "random",  # grid, random, bayes
    "metric": {
        "name": "mean_steps",
        "goal": "maximize",
    },
    "parameters": {
        "gamma": {"values": [0.95, 0.99, 0.995, 0.999]},
        "batch_count": {"values": [1, 4, 8, 16]},
        "entropy_coeff": {"values": [0.0, 0.01, 0.1, 0.5]},
        "actor_noise_std": {"values": [0.0, 0.1, 0.3, 0.5]},
        "learning_rate": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
        "schedule": {"values": ["fixed", "adaptive"]},
        "value_coeff": {"values": [0.5, 1.0, 1.5, 2.0]},
        "clip_ratio": {"values": [0.1, 0.2, 0.3]},
        "target_kl": {"values": [0.01, 0.02, 0.03]},
        "vel_weight": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "torque_weight": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "action_diff_weight": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "x_diff_weight": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "x_reward_weight": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "diff_term_weight": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "termination_reward": {"values": [0.0, 1.0, 10.0, 100.0]},
    },
}


def make_wandb_eval_cb(run):
    def cb(runner, stat):
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


def train():
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        "gamma": 0.99,
        "batch_count": 4,
        "entropy_coeff": 0.0,
        "actor_noise_std": 0.3,
        "learning_rate": 1e-3,
        "schedule": "fixed",
        "value_coeff": 1.0,
        "clip_ratio": 0.2,
        "target_kl": 0.01,
        "vel_weight": 1e-3,
        "torque_weight": 1e-3,
        "action_diff_weight": 1e-3,
        "x_diff_weight": 1e-3,
        "x_reward_weight": 1e-3,
        "diff_term_weight": 1e-3,
        "termination_reward": 10.0,
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    reward_fun = get_reward_fun(
        [
            config.vel_weight,
            config.torque_weight,
            config.action_diff_weight,
            config.x_diff_weight,
            config.x_reward_weight,
            config.diff_term_weight,
        ]
    )

    aio_cfg = AIOlympicsCfg(
        num_envs=10_000,
        seed=42,
        max_episode_length_time=5.0,
        robot=Robot.ACROBOT,
        dt=0.002,
        iterations=50,
    )
    torch.manual_seed(aio_cfg.seed)

    actor_layers = [32] * 2
    actor_activations = ["selu"] * 2 + ["linear"]
    critic_layers = [32] * 2
    critic_activations = ["selu"] * 2 + ["linear"]

    agent_cfg = AIOlympicsPPOCfg(
        batch_count=config.batch_count,
        gamma=config.gamma,
        actor_activations=actor_activations,
        critic_activations=critic_activations,
        actor_hidden_dims=actor_layers,
        critic_hidden_dims=critic_layers,
        actor_noise_std=config.actor_noise_std,
        learning_rate=config.learning_rate,
        schedule=config.schedule,
        clip_ratio=config.clip_ratio,
        entropy_coeff=config.entropy_coeff,
        gae_lambda=0.97,
        target_kl=config.target_kl,
        value_coeff=config.value_coeff,
    )

    runner_cfg = AIOlympicsRunnerCfg(num_steps_per_env=100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = aio_cfg.num_envs

    # simulation parameter
    max_episode_length = aio_cfg.max_episode_length_steps
    integrator = Integrator.RUNGE_KUTTA
    plant_params = load_plant_params(num_envs, aio_cfg.dt, device)

    env_kwargs = dict(
        num_envs=num_envs,
        device=device,
        max_episode_length=max_episode_length,
        plant_params=plant_params,
        integrator=integrator,
        reset_fun=reset_fun,
        reward_fun=reward_fun,
        termination_reward=-config.termination_reward,
    )
    env = TorchEnv(**env_kwargs)

    agent = PPO(env, device=device, **asdict(agent_cfg))

    config = dict(
        agent_kwargs=asdict(agent_cfg),
        # env_kwargs=env_kwargs,
        runner_kwargs=asdict(runner_cfg),
    )
    wandb_learn_config = dict(
        config=config,
        entity=WANDB_ENTITY,
        project="ai_olympics",
    )
    run = wandb.init(**wandb_learn_config)

    log_root_path = os.path.join("logs", "ai_olympics")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    learn_cb = [
        Runner._log,
        make_interval_cb(make_save_model_cb(log_dir), interval=aio_cfg.iterations),
    ]
    eval_cb = [
        make_wandb_eval_cb(run),
    ]

    runner = Runner(
        env,
        agent,
        device=device,
        learn_cb=learn_cb,
        evaluation_cb=eval_cb,
        **asdict(runner_cfg),
    )

    runner.learn(iterations=aio_cfg.iterations, return_epochs=10)
    print("Running evaluation")
    runner.evaluate(aio_cfg.max_episode_length_steps, return_epochs=10)

    run.finish()


def main():
    # sweep_id = wandb.sweep(
    #     sweep_config, entity="wojcikmichal98-none", project="ai_olympics"
    # )
    sweep_id = "aaq89z1m"
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, train, entity="wojcikmichal98-none", project="ai_olympics")


if __name__ == "__main__":
    main()
