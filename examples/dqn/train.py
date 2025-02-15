import os
from dataclasses import asdict
from datetime import datetime

import torch
import wandb
from callbacks import (
    make_interval_cb,
    make_save_model_cb,
    make_wandb_cb,
    make_wandb_eval_cb,
)
from double_pendulum.model.torch_plant import Integrator
from double_pendulum.simulation.torch_env import TorchEnv
from dqn import DQN
from mdp import reset_fun, reward_fun
from utils import load_plant_params, parse_args
from wandb_config import WANDB_API_KEY, WANDB_ENTITY

os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def main():
    aio_cfg, dqn_cfg = parse_args()
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
    print(f"max_episode_length: {max_episode_length}")
    env = TorchEnv(**env_kwargs)

    config = dict(
        agent_kwargs=asdict(dqn_cfg),
        # env_kwargs=env_kwargs,
    )
    wandb_learn_config = dict(
        config=config,
        entity=WANDB_ENTITY,
        project="ai_olympics",
    )
    if aio_cfg.log_wandb:
        run = wandb.init(**wandb_learn_config)

    log_root_path = os.path.join("logs", "ai_olympics")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    train_cbs = [
        make_interval_cb(make_save_model_cb(log_dir), interval=50),
        DQN._log,
    ]
    eval_cbs = []
    if aio_cfg.log_wandb:
        train_cbs.append(make_wandb_cb(run))
        eval_cbs.append(make_wandb_eval_cb(run))

    dqn = DQN(
        env,
        device,
        train_cbs=train_cbs,
        evaluation_cbs=eval_cbs,
        **asdict(dqn_cfg),
    )
    dqn.train(
        num_iterations=aio_cfg.iterations,
        num_steps_per_env=dqn_cfg.num_steps_per_env,
    )

    print("\n\nRunning evaluation\n\n")
    dqn.eval(
        num_steps=aio_cfg.max_episode_length_steps,
        return_epochs=10,
    )

    if aio_cfg.log_wandb:
        run.finish()


if __name__ == "__main__":
    main()
