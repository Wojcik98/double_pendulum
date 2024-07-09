import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.torch_plant import Integrator, PlantParams
from double_pendulum.simulation.torch_env import TorchEnv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    num_iters = 1_000
    num_envs = 10
    design = "design_C.0"
    model = "model_3.0"
    traj_model = "model_3.1"
    robot = "acrobot"

    if robot == "acrobot":
        torque_limit = [0.0, 6.0]
    if robot == "pendubot":
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

    # simulation parameter
    dt = 0.005
    t_final = 10.0  # 5.985
    max_episode_length = int(t_final / dt)
    integrator = Integrator.RUNGE_KUTTA
    # start = [0.0, 0.0, 0.0, 0.0]
    # goal = [np.pi, 0.0, 0.0, 0.0]

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

    def reset_fun(num_envs: int, device: torch.device) -> torch.Tensor:
        mean = torch.zeros(num_envs, 4, device=device)
        std = 1.0
        return torch.normal(mean, std)

    env = TorchEnv(
        num_envs,
        device,
        max_episode_length=max_episode_length,
        plant_params=params,
        integrator=integrator,
        reset_fun=reset_fun,
    )

    actions = torch.zeros((num_envs, 1), device=device)
    posx = []
    posy = []

    for _ in range(num_iters):
        state, reward, done, info = env.step(actions)
        kin = env.plant.forward_kinematics(state[:, :2])[0, :].cpu().numpy()
        posx.append([0.0, kin[0, 0], kin[1, 0]])
        posy.append([0.0, kin[0, 1], kin[1, 1]])

    print("Done simulating")

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
    plt.show()


if __name__ == "__main__":
    main()
