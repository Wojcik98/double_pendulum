import os
from dataclasses import asdict
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.torch_plant import Integrator, PlantParams, TorchPlant
from double_pendulum.simulation.torch_env import Robot, TorchEnv
from rsl_rl.runners import OnPolicyRunner
from utils import load_plant_params, parse_args, get_checkpoint_path


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
        # mean[:, 0] = torch.pi
        std = 0.5
        return torch.normal(mean, std)

    def reward_fun(
        state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        kin = plant.forward_kinematics(state[:, :2])
        ee_y = kin[:, 1, 1]
        # print(ee_y.mean())
        vel = state[:, 2:].abs().sum(dim=1)

        reward = ee_y - 0.001 * vel

        return reward

    env = TorchEnv(
        num_envs,
        device,
        max_episode_length=max_episode_length,
        plant_params=plant_params,
        integrator=integrator,
        reset_fun=reset_fun,
        reward_fun=reward_fun,
    )

    log_root_path = os.path.join("logs", rl_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, rl_cfg.load_run, rl_cfg.load_checkpoint
    )
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if rl_cfg.run_name:
        log_dir += f"_{rl_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    runner = OnPolicyRunner(env, asdict(rl_cfg), log_dir=log_dir, device=device)
    runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    policy = runner.get_inference_policy(device=device)
    obs, _ = env.get_observations()
    posx = []
    posy = []

    for _ in range(max_episode_length):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            kin = env.plant.forward_kinematics(obs[:, :2])[0, :].cpu().numpy()
            posx.append([0.0, kin[0, 0], kin[1, 0]])
            posy.append([0.0, kin[0, 1], kin[1, 1]])

    # plt.ion()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))
    ax.set_aspect("equal")
    ax.grid()

    (line,) = ax.plot([], [], "o-", lw=2)
    (trace,) = ax.plot([], [], ".-", lw=1, ms=2)
    time_template = "time = %.1fs"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def animate(i):
        thisx = posx[i]
        thisy = posy[i]

        history_x = [x[2] for x in posx[:i]]
        history_y = [y[2] for y in posy[:i]]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i * dt))
        return line, trace, time_text

    ani = animation.FuncAnimation(fig, animate, len(posx), interval=dt * 100, blit=True)
    # ani.save(f"{rl_cfg.load_run}.mp4")
    plt.show()


if __name__ == "__main__":
    main()
