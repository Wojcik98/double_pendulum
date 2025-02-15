import os
from dataclasses import asdict
from datetime import datetime

import torch
from double_pendulum.model.torch_plant import Integrator
from double_pendulum.simulation.buffer_env import BufferEnv
from mdp import reset_fun, reward_fun
from rsl_rl.algorithms import PPO, SAC
from rsl_rl.runners import Runner
from rsl_rl.runners.callbacks import make_interval_cb, make_save_model_cb, make_wandb_cb
from utils import load_plant_params, parse_args
from wandb_config import WANDB_API_KEY, WANDB_ENTITY

os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def main():
    aio_cfg, agent_cfg, runner_cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    num_envs = aio_cfg.num_envs
    print(f"Number of environments: {num_envs}")

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
    env = BufferEnv(gamma=agent_cfg.gamma, **env_kwargs)

    agent = SAC(env, device=device, **asdict(agent_cfg))
    # agent = PPO(env, device=device)

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
        # make_wandb_cb(wandb_learn_config),
        make_interval_cb(make_save_model_cb(log_dir), interval=50),
    ]

    runner.learn(iterations=aio_cfg.iterations, return_epochs=100)


if __name__ == "__main__":
    main()
