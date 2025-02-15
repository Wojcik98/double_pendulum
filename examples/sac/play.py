import os
from dataclasses import asdict
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.torch_plant import Integrator, PlantParams, TorchPlant
from double_pendulum.simulation.torch_env import Robot, TorchEnv
from mdp import reset_fun, reward_fun
from rsl_rl.algorithms import PPO, SAC
from rsl_rl.runners import Runner
from utils import get_checkpoint_path, load_plant_params, parse_args


def main():
    aio_cfg, agent_cfg, runner_cfg = parse_args()
    aio_cfg.num_envs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    num_envs = aio_cfg.num_envs

    # simulation parameter
    dt = aio_cfg.dt
    max_episode_length = aio_cfg.max_episode_length_steps
    integrator = Integrator.RUNGE_KUTTA
    plant_params = load_plant_params(num_envs, dt, device)

    env = TorchEnv(
        num_envs,
        device,
        max_episode_length=max_episode_length,
        plant_params=plant_params,
        integrator=integrator,
        reset_fun=reset_fun,
        reward_fun=reward_fun,
    )

    # agent = SAC(env, device=device, **asdict(agent_cfg))
    agent = PPO(
        env,
        device=device,
        batch_size=agent_cfg.batch_size,
        batch_count=agent_cfg.batch_count,
        gamma=agent_cfg.gamma,
        actor_activations=agent_cfg.actor_activations,
        critic_activations=agent_cfg.critic_activations,
        actor_hidden_dims=agent_cfg.actor_hidden_dims,
        critic_hidden_dims=agent_cfg.critic_hidden_dims,
        action_min=agent_cfg.action_min,
        action_max=agent_cfg.action_max,
    )

    log_root_path = os.path.join("logs", "ai_olympics")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(
        log_root_path, aio_cfg.load_run, aio_cfg.checkpoint
    )
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    runner = Runner(env, agent, device=device, **asdict(runner_cfg))
    runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    policy = runner.get_inference_policy(device=device)
    obs, _ = env.get_observations()
    posx = []
    posy = []

    for _ in range(max_episode_length):
        with torch.inference_mode():
            actions = policy(obs)
            # print(actions.abs().max())
            # critic_input = agent._critic_input(obs, actions)
            # q1 = agent.critic_1(critic_input)
            # q2 = agent.critic_2(critic_input)
            # q = torch.min(q1, q2)
            # print(f"Q1: {q1}, Q2: {q2}")
            obs, rewards, dones, info = env.step(actions)
            real_obs = env._denormalize_state(obs)
            kin = env.plant.forward_kinematics(real_obs[:, :2])[0, :].cpu().numpy()
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
