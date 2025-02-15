import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime

import torch
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.torch_plant import Integrator, PlantParams
from double_pendulum.simulation.torch_env import Robot, TorchEnv
from rl_cfg import AIOlympicsDQNCfg
from rsl_rl_cfg_defs import DQNCfg


@dataclass
class AIOlympicsCfg:
    num_envs: int
    seed: int
    max_episode_length_time: float
    robot: Robot
    dt: float = 0.002
    iterations: int = 100
    load_run: str = ""
    checkpoint: str = ".*"
    log_wandb: bool = False

    @property
    def max_episode_length_steps(self) -> int:
        return int(self.max_episode_length_time / self.dt)


def parse_args() -> tuple[AIOlympicsCfg, DQNCfg]:
    args_cli = get_cli()
    if args_cli.robot == "acrobot":
        robot = Robot.ACROBOT
    else:
        robot = Robot.PENDUBOT

    aio_cfg = AIOlympicsCfg(
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        max_episode_length_time=args_cli.max_episode_length_time,
        robot=robot,
        dt=args_cli.dt,
        iterations=args_cli.iterations,
        log_wandb=args_cli.log,
    )

    rl_cfg = AIOlympicsDQNCfg()

    if args_cli.num_steps_per_env is not None:
        rl_cfg.num_steps_per_env = args_cli.num_steps_per_env

    return aio_cfg, rl_cfg


def get_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_envs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_length_time", type=float, default=5.0)
    parser.add_argument("--robot", type=str, default="acrobot")
    parser.add_argument("--dt", type=float, default=0.002)

    arg_group = parser.add_argument_group(
        "rsl_rl", description="Arguments for RSL-RL agent."
    )
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix to the log directory.",
    )
    arg_group.add_argument(
        "--num_steps_per_env",
        type=int,
        default=None,
        help="num_steps_per_env",
    )
    arg_group.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="iterations",
    )
    # -- load arguments
    arg_group.add_argument(
        "--resume", type=bool, default=None, help="Whether to resume from a checkpoint."
    )
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run folder to resume from.",
    )
    arg_group.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint file to resume from."
    )
    # -- logger arguments
    arg_group.add_argument(
        "--log",
        default=False,
        help="Whether to log the experiment using wandb.",
        action="store_true",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )

    args_cli = parser.parse_args()
    return args_cli


def load_plant_params(num_envs: int, dt: float, device: torch.device) -> PlantParams:
    design = "design_C.0"
    model = "model_3.0"
    robot = Robot.ACROBOT

    if robot == Robot.ACROBOT:
        torque_limit = [0.0, 6.0]
    if robot == Robot.PENDUBOT:
        torque_limit = [6.0, 0.0]

    model_par_path = (
        "../../data/system_identification/identified_parameters/"
        + design
        + "/"
        + model
        + "/model_parameters.yml"
    )
    mpar = model_parameters(filepath=model_par_path)
    mpar.set_motor_inertia(0.0)
    mpar.set_damping([0.0, 0.0])
    mpar.set_cfric([0.0, 0.0])
    mpar.set_torque_limit(torque_limit)

    params = PlantParams(
        mass=torch.tensor([mpar.m] * num_envs, device=device),
        length=torch.tensor([mpar.l] * num_envs, device=device),
        com=torch.tensor([mpar.r] * num_envs, device=device),
        damping=torch.tensor([mpar.b] * num_envs, device=device),
        gravity=torch.tensor([mpar.g] * num_envs, device=device),
        coulomb_fric=torch.tensor([mpar.cf] * num_envs, device=device),
        inertia=torch.tensor([mpar.I] * num_envs, device=device),
        motor_inertia=torch.tensor([mpar.Ir] * num_envs, device=device),
        gear_ratio=torch.tensor([mpar.gr] * num_envs, device=device),
        torque_limit=torch.tensor([mpar.tl] * num_envs, device=device),
        dt=dt,
    )

    return params


def get_checkpoint_path(
    log_path: str,
    run_dir: str = ".*",
    checkpoint: str = ".*",
    other_dirs: list[str] = None,
    sort_alpha: bool = True,
) -> str:
    """Get path to the model checkpoint in input directory.

    The checkpoint file is resolved as: ``<log_path>/<run_dir>/<*other_dirs>/<checkpoint>``, where the
    :attr:`other_dirs` are intermediate folder names to concatenate. These cannot be regex expressions.

    If :attr:`run_dir` and :attr:`checkpoint` are regex expressions then the most recent (highest alphabetical order)
    run and checkpoint are selected. To disable this behavior, set the flag :attr:`sort_alpha` to False.

    Args:
        log_path: The log directory path to find models in.
        run_dir: The regex expression for the name of the directory containing the run. Defaults to the most
            recent directory created inside :attr:`log_path`.
        other_dirs: The intermediate directories between the run directory and the checkpoint file. Defaults to
            None, which implies that checkpoint file is directly under the run directory.
        checkpoint: The regex expression for the model checkpoint file. Defaults to the most recent
            torch-model saved in the :attr:`run_dir` directory.
        sort_alpha: Whether to sort the runs by alphabetical order. Defaults to True.
            If False, the folders in :attr:`run_dir` are sorted by the last modified time.

    Raises:
        ValueError: When no runs are found in the input directory.
        ValueError: When no checkpoints are found in the input directory.

    Returns:
        The path to the model checkpoint.
    """
    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [
            os.path.join(log_path, run)
            for run in os.scandir(log_path)
            if run.is_dir() and re.match(run_dir, run.name)
        ]
        # sort matched runs by alphabetical order (latest run should be last)
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        if other_dirs is not None:
            run_path = os.path.join(runs[-1], *other_dirs)
        else:
            run_path = runs[-1]
    except IndexError:
        raise ValueError(
            f"No runs present in the directory: '{log_path}' match: '{run_dir}'."
        )

    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(
            f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'."
        )
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(run_path, checkpoint_file)
