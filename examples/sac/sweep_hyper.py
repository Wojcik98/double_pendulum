import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
import wandb
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.torch_plant import Integrator, PlantParams, TorchPlant
from double_pendulum.simulation.torch_env import Robot, TorchEnv
from mdp import reset_fun, reward_fun
from rl_cfg import AIOlympicsRunnerCfg, AIOlympicsSACCfg
from rsl_rl.algorithms import SAC
from rsl_rl.runners import Runner
from rsl_rl.runners.callbacks import make_interval_cb, make_save_model_cb, make_wandb_cb
from utils import AIOlympicsCfg, load_plant_params
from wandb_config import WANDB_API_KEY, WANDB_ENTITY

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

sweep_config = {
    "method": "bayes",  # grid, random, bayes
    "metric": {
        "name": "mean_steps",
        "goal": "maximize",
    },
    "parameters": {
        "gamma": {"values": [0.8, 0.9, 0.95, 0.99, 0.995, 0.999]},
        "alpha": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "alpha_lr": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
        "actor_lr": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
        "critic_lr": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
        "polyak": {"values": [0.8, 0.9, 0.95, 0.99, 0.995, 0.999]},
        "return_steps": {"values": [1, 2, 3, 4]},
        "batch_size": {"values": [1, 2, 4, 8, 16]},
        "batch_count": {"values": [10, 20, 40, 80, 160]},
        "gradient_clip": {"values": [0.5, 1.0, 1.5, 2.0]},
        "num_steps_per_env": {"values": [100, 250, 500, 1000]},
    },
}


def train():
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        "alpha": 0.2,
        "alpha_lr": 1e-3,
        "gamma": 0.99,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "polyak": 0.995,
        "return_steps": 1,
        "batch_size": 1,
        "batch_count": 10,
        "gradient_clip": 1.0,
        "num_steps_per_env": 250,
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    aio_cfg = AIOlympicsCfg(
        num_envs=10_000,
        seed=42,
        max_episode_length_time=5.0,
        robot=Robot.ACROBOT,
        dt=0.002,
        iterations=100,
    )
    torch.manual_seed(aio_cfg.seed)

    actor_layers = [32] * 2
    actor_activations = ["relu"] * 2 + ["linear"]
    critic_layers = [32] * 2
    critic_activations = ["relu"] * 2 + ["linear"]

    agent_cfg = AIOlympicsSACCfg(
        alpha=config.alpha,
        alpha_lr=config.alpha_lr,
        gamma=config.gamma,
        actor_activations=actor_activations,
        actor_hidden_dims=actor_layers,
        actor_init_gain=0.5,
        actor_input_normalization=False,
        actor_noise_std=0.2,
        actor_lr=config.actor_lr,
        chimera=True,
        critic_activations=critic_activations,
        critic_hidden_dims=critic_layers,
        critic_init_gain=0.5,
        critic_input_normalization=False,
        critic_lr=config.critic_lr,
        polyak=config.polyak,
        return_steps=config.return_steps,
        batch_size=config.batch_size,
        batch_count=config.batch_count,
        gradient_clip=config.gradient_clip,
        storage_size=10_000_000,
    )

    runner_cfg = AIOlympicsRunnerCfg(num_steps_per_env=config.num_steps_per_env)

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
        termination_reward=0.0,
    )
    env = TorchEnv(**env_kwargs)

    agent = SAC(env, device=device, **asdict(agent_cfg))

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

    log_root_path = os.path.join("logs", "ai_olympics")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    runner = Runner(env, agent, device=device, **asdict(runner_cfg))
    runner._learn_cb = [
        Runner._log,
        make_wandb_cb(wandb_learn_config),
        make_interval_cb(make_save_model_cb(log_dir), interval=50),
    ]

    runner.learn(iterations=aio_cfg.iterations, return_epochs=100)


def main():
    sweep_id = wandb.sweep(
        sweep_config, entity="wojcikmichal98-none", project="ai_olympics"
    )
    # sweep_id = "qxllbqhp"
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, train, entity="wojcikmichal98-none", project="ai_olympics")


if __name__ == "__main__":
    main()
