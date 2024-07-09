from dataclasses import dataclass
from enum import Enum, auto

import torch


def bmv(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matrix-vector multiplication.

    Params:
        A - (batch, n, n) tensor
        b - (batch, n) tensor

    Returns:
        (batch, n) tensor
    """
    return torch.bmm(A, b.unsqueeze(dim=2)).squeeze()


class Integrator(Enum):
    RUNGE_KUTTA = auto()
    EULER = auto()


@dataclass
class PlantParams:
    """
    mass : torch.Tensor
        shape=(num_envs, 2), dtype=float
        masses of the double pendulum,
        [m1, m2], units=[kg]
    length : torch.Tensor
        shape=(num_envs, 2), dtype=float
        link lengths of the double pendulum,
        [l1, l2], units=[m]
    com : torch.Tensor
        shape=(num_envs, 2), dtype=float
        center of mass lengths of the double pendulum links
        [r1, r2], units=[m]
    damping : torch.Tensor
        shape=(num_envs, 2), dtype=float
        damping coefficients of the double pendulum actuators
        [b1, b2], units=[kg*m/s]
    gravity : torch.Tensor
        shape=(num_envs,), dtype=float
        gravity acceleration (pointing downwards),
        units=[m/s²]
    coulomb_fric : torch.Tensor
        shape=(num_envs, 2), dtype=float
        coulomb friction coefficients for the double pendulum actuators
        [cf1, cf2], units=[Nm]
    inertia : torch.Tensor
        shape=(num_envs, 2), dtype=float
        inertia of the double pendulum links
        [I1, I2], units=[kg*m²]
        if entry is None defaults to point mass m*l² inertia for the entry
    motor_inertia : torch.Tensor
        shape=(num_envs,), dtype=float
        inertia of the actuators/motors
        units=[kg*m²]
    gear_ratio : torch.Tensor
        shape=(num_envs,), dtype=int
        gear ratio of the motors
    torque_limit : torch.Tensor
        shape=(num_envs, 2), dtype=float
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    dt: step size
        float
    """

    mass: torch.Tensor
    length: torch.Tensor
    com: torch.Tensor
    damping: torch.Tensor
    gravity: torch.Tensor
    coulomb_fric: torch.Tensor
    inertia: torch.Tensor
    motor_inertia: torch.Tensor
    gear_ratio: torch.Tensor
    torque_limit: torch.Tensor
    dt: float


class TorchPlant:
    """Double pendulum plant implementation using PyTorch."""

    def __init__(
        self,
        params: PlantParams,
    ):
        self.num_envs = params.mass.shape[0]
        self.device = params.mass.device

        self.m = params.mass
        self.l = params.length
        self.com = params.com
        self.b = params.damping
        self.g = params.gravity
        self.coulomb_fric = params.coulomb_fric
        self.I = params.inertia
        self.Ir = params.motor_inertia
        self.gr = params.gear_ratio
        self.torque_limit = params.torque_limit

        self.dof = 2

        if self.torque_limit[0, 0] == 0:
            B_tile = torch.tensor(
                [[0, 0], [0, 1]], dtype=torch.float, device=self.device
            )
        elif self.torque_limit[0, 1] == 0:
            B_tile = torch.tensor(
                [[1, 0], [0, 0]], dtype=torch.float, device=self.device
            )
        else:
            B_tile = torch.tensor(
                [[1, 0], [0, 1]], dtype=torch.float, device=self.device
            )

        self.B = torch.tile(B_tile, (self.num_envs, 1, 1))

    def forward_kinematics(self, pos: torch.Tensor):
        """
        Forward kinematics, origin at fixed point.

        Parameters
        ----------
        pos : torch.Tensor, shape=(num_envs, 2), dtype=float,
            positions of the double pendulum,
            order=[angle1, angle2],
            units=[rad]

        Returns
        -------
        torch.Tensor, shape=(num_envs, 2, 2), [[x1, y1], [x2, y2]]
            cartesian coordinates of the link end points
            units=[m]
        """
        ee1_pos_x = self.l[:, 0] * torch.sin(pos[:, 0])
        ee1_pos_y = -self.l[:, 0] * torch.cos(pos[:, 0])

        ee2_pos_x = ee1_pos_x + self.l[:, 1] * torch.sin(pos[:, 0] + pos[:, 1])
        ee2_pos_y = ee1_pos_y - self.l[:, 1] * torch.cos(pos[:, 0] + pos[:, 1])

        K = torch.zeros((self.num_envs, self.dof, self.dof), device=self.device)
        K[:, 0, 0] = ee1_pos_x
        K[:, 0, 1] = ee1_pos_y
        K[:, 1, 0] = ee2_pos_x
        K[:, 1, 1] = ee2_pos_y

        return K

    def mass_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mass matrix from the equations of motion.

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        torch.Tensor, shape=(num_envs, 2, 2),
            mass matrix
        """
        pos = torch.clone(x[:, : self.dof])

        m00 = (
            self.I[:, 0]
            + self.I[:, 1]
            + self.m[:, 1] * self.l[:, 0] ** 2.0
            + 2 * self.m[:, 1] * self.l[:, 0] * self.com[:, 1] * torch.cos(pos[:, 1])
            + self.gr**2.0 * self.Ir
            + self.Ir
        )
        m01 = (
            self.I[:, 1]
            + self.m[:, 1] * self.l[:, 0] * self.com[:, 1] * torch.cos(pos[:, 1])
            - self.gr * self.Ir
        )
        m10 = (
            self.I[:, 1]
            + self.m[:, 1] * self.l[:, 0] * self.com[:, 1] * torch.cos(pos[:, 1])
            - self.gr * self.Ir
        )
        m11 = self.I[:, 1] + self.gr**2.0 * self.Ir
        M = torch.zeros((self.num_envs, self.dof, self.dof), device=self.device)
        M[:, 0, 0] = m00
        M[:, 0, 1] = m01
        M[:, 1, 0] = m10
        M[:, 1, 1] = m11

        return M

    def coriolis_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Coriolis matrix from the equations of motion.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]


        Returns
        -------
        torch.Tensor, shape=(num_envs, 2, 2),
            coriolis matrix
        """
        pos = torch.clone(x[:, : self.dof])
        vel = torch.clone(x[:, self.dof :])

        C00 = (
            -2
            * self.m[:, 1]
            * self.l[:, 0]
            * self.com[:, 1]
            * torch.sin(pos[:, 1])
            * vel[:, 1]
        )
        C01 = (
            -self.m[:, 1]
            * self.l[:, 0]
            * self.com[:, 1]
            * torch.sin(pos[:, 1])
            * vel[:, 1]
        )
        C10 = (
            self.m[:, 1]
            * self.l[:, 0]
            * self.com[:, 1]
            * torch.sin(pos[:, 1])
            * vel[:, 0]
        )
        C11 = 0
        C = torch.zeros((self.num_envs, self.dof, self.dof), device=self.device)
        C[:, 0, 0] = C00
        C[:, 0, 1] = C01
        C[:, 1, 0] = C10
        C[:, 1, 1] = C11

        return C

    def gravity_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        gravity vector from the equations of motion

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        torch.Tensor, shape=(num_envs, 2),
            gravity vector
        """
        pos = torch.clone(x[:, : self.dof])

        G0 = -self.m[:, 0] * self.g * self.com[:, 0] * torch.sin(pos[:, 0]) - self.m[
            :, 1
        ] * self.g * (
            self.l[:, 0] * torch.sin(pos[:, 0])
            + self.com[:, 1] * torch.sin(pos[:, 0] + pos[:, 1])
        )
        G1 = -self.m[:, 1] * self.g * self.com[:, 1] * torch.sin(pos[:, 0] + pos[:, 1])
        G = torch.stack((G0, G1), dim=1)

        return G

    def coulomb_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        coulomb vector from the equations of motion

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        torch.Tensor, shape=(num_envs, 2),
            coulomb vector

        """
        vel = torch.clone(x[:, self.dof :])

        F = torch.zeros((self.num_envs, self.dof), device=self.device)
        F[:, :] = self.b[:, :] * vel[:, :] + self.coulomb_fric[:, :] * torch.arctan(
            100 * vel[:, :]
        )

        return F

    def kinetic_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        kinetic energy of the double pendulum

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        torch.Tensor, shape=(num_envs,),
            kinetic energy, units=[J]
        """
        vel = torch.clone(x[:, self.dof :])

        M = self.mass_matrix(x)
        kin = torch.bmm(M, vel.unsqueeze(dim=2)).squeeze()
        # kin = 0.5 *  torch.dot(vel, kin)
        kin = 0.5 * torch.bmm(vel.unsqueeze(dim=1), kin.unsqueeze(dim=2).squeeze())

        return kin

    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        potential energy of the double pendulum

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        torch.Tensor, shape=(num_envs,),
            potential energy, units=[J]
        """
        pos = torch.clone(x[:, : self.dof])

        # 0 level at hinge
        y1 = -self.com[:, 0] * torch.cos(pos[:, 0])
        y2 = -self.l[:, 0] * torch.cos(pos[:, 0]) - self.com[:, 1] * torch.cos(
            pos[:, 1] + pos[:, 0]
        )
        pot = self.m[:, 0] * self.g * y1 + self.m[:, 1] * self.g * y2

        return pot

    def total_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        total energy of the double pendulum

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        Returns
        -------
        torch.Tensor, shape=(num_envs,),
            total energy, units=[J]
        """
        E = self.kinetic_energy(x) + self.potential_energy(x)
        return E

    def forward_dynamics(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        forward dynamics of the double pendulum

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        tau : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        torch.Tensor, shape=(num_envs, 2),
            joint acceleration, [acc1, acc2], units=[m/s²]
        """
        vel = torch.clone(x[:, self.dof :])

        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        B_dot_tau = torch.bmm(self.B, tau.unsqueeze(dim=2)).squeeze()
        C_dot_vel = torch.bmm(C, vel.unsqueeze(dim=2)).squeeze()

        force = G + B_dot_tau - C_dot_vel
        friction = F
        diff = force - friction

        accn = torch.linalg.solve(M, diff)
        return accn

    def rhs(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        integrand of the equations of motion

        Parameters
        ----------
        state : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        tau : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        torch.Tensor, shape=(num_envs, 4),
            integrand, [vel1, vel2, acc1, acc2]
        """
        # Forward dynamics
        accn = self.forward_dynamics(state, tau)

        # Next state
        res = torch.zeros((self.num_envs, 2 * self.dof), device=self.device)
        res[:, 0] = state[:, 2]
        res[:, 1] = state[:, 3]
        res[:, 2] = accn[:, 0]
        res[:, 3] = accn[:, 1]

        return res

    def get_Mx(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        state derivative of mass matrix

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        torch.Tensor, shape=(num_envs, 4, 2, 2),
            derivative of mass matrix,
            Mx[i]=del(M)/del(x_i)
        """
        Mx = torch.zeros(
            (self.num_envs, 2 * self.dof, self.dof, self.dof), device=self.device
        )

        Mx[:, 1, 0, 0] = (
            -2 * self.l[:, 0] * self.m[:, 1] * self.com[:, 1] * torch.sin(x[:, 1])
        )
        Mx[:, 1, 0, 1] = (
            -self.l[:, 0] * self.m[:, 1] * self.com[:, 1] * torch.sin(x[:, 1])
        )
        Mx[:, 1, 1, 0] = (
            -self.l[:, 0] * self.m[:, 1] * self.com[:, 1] * torch.sin(x[:, 1])
        )
        Mx[:, 1, 1, 1] = 0

        return Mx

    def get_Minvx(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        state derivative of inverse mass matrix

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        torch.Tensor, shape=(num_envs, 4, 2, 2),
            derivative of inverse mass matrix,
            Minvx[i]=del(Minv)/del(x_i)
        """
        Minvx = torch.zeros(
            (self.num_envs, 2 * self.dof, self.dof, self.dof), device=self.device
        )

        den = (
            -self.I[:, 0] * self.I[:, 1]
            - self.I[:, 1] * self.l[:, 0] ** 2.0 * self.m[:, 1]
            + (self.l[:, 0] * self.m[:, 1] * self.com[:, 1] * torch.cos(x[:, 1])) ** 2.0
        )

        h1 = self.l[:, 0] * self.m[:, 1] * self.com[:, 1]

        Minvx[:, 1, 0, 0] = (
            -2.0
            * self.I[:, 1]
            * h1**2.0
            * torch.sin(x[:, 1])
            * torch.cos(x[:, 1])
            / den**2.0
        )
        Minvx[:, 1, 0, 1] = (
            2
            * h1**2.0
            * (self.I[:, 1] + h1 * torch.cos(x[:, 1]))
            * torch.cos(x[:, 1])
            * torch.sin(x[:, 1])
            / den**2.0
            - h1 * torch.sin(x[:, 1]) / den
        )
        Minvx[:, 1, 1, 0] = (
            2
            * h1**2.0
            * (self.I[:, 1] + h1 * torch.cos(x[:, 1]))
            * torch.cos(x[:, 1])
            * torch.sin(x[:, 1])
            / den**2.0
            - h1 * torch.sin(x[:, 1]) / den
        )
        Minvx[:, 1, 1, 1] = (
            2
            * h1**2
            * (
                -self.I[:, 0]
                - self.I[:, 1]
                - self.l[:, 0] ** 2.0 * self.m[:, 1]
                - 2 * h1 * torch.cos(x[:, 1])
            )
            * torch.cos(x[:, 1])
            * torch.sin(x[:, 1])
            / den**2.0
            + 2 * h1 * torch.sin(x[:, 1]) / den
        )

        return Minvx

    def get_Cx(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        state derivative of coriolis matrix

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        torch.Tensor, shape=(num_envs, 4, 2, 2),
            derivative of coriolis matrix,
            Cx[i]=del(C)/del(x_i)
        """
        Cx = torch.zeros(
            (self.num_envs, 2 * self.dof, self.dof, self.dof), device=self.device
        )

        h1 = self.l[:, 0] * self.m[:, 1] * self.com[:, 1]

        Cx[:, 1, 0, 0] = -2.0 * h1 * torch.cos(x[:, 1]) * x[:, 3]
        Cx[:, 1, 0, 1] = -h1 * torch.cos(x[:, 1]) * x[:, 3]
        Cx[:, 1, 1, 0] = h1 * torch.cos(x[:, 1]) * x[:, 2]

        Cx[:, 2, 1, 0] = h1 * torch.sin(x[:, 1])

        Cx[:, 3, 0, 0] = -2 * h1 * torch.sin(x[:, 1])
        Cx[:, 3, 0, 1] = -h1 * torch.sin(x[:, 1])

        return Cx

    def get_Gx(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        state derivative of gravity vector

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        torch.Tensor, shape=(num_envs, 2, 4),
            derivative of gravity vector,
            Gx[:, i]=del(G)/del(x_i)
        """
        Gx = torch.zeros((self.num_envs, self.dof, 2 * self.dof), device=self.device)

        Gx[:, 0, 0] = -self.g * self.m[:, 0] * self.com[:, 0] * torch.cos(
            x[:, 0]
        ) - self.g * self.m[:, 1] * (
            self.l[:, 0] * torch.cos(x[:, 0])
            + self.com[:, 1] * torch.cos(x[:, 0] + x[:, 1])
        )
        Gx[:, 0, 1] = (
            -self.g * self.m[:, 1] * self.com[:, 1] * torch.cos(x[:, 0] + x[:, 1])
        )

        Gx[:, 1, 0] = (
            -self.g * self.m[:, 1] * self.com[:, 1] * torch.cos(x[:, 0] + x[:, 1])
        )
        Gx[:, 1, 1] = (
            -self.g * self.m[:, 1] * self.com[:, 1] * torch.cos(x[:, 0] + x[:, 1])
        )

        return Gx

    def get_Fx(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        state derivative of coulomb vector

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
            not used

        Returns
        -------
        numpy array
        torch.Tensor, shape=(num_envs, 2, 4),
            derivative of coulomb vector,
            Fx[:, i]=del(F)/del(x_i)

        """
        Fx = torch.zeros((self.num_envs, self.dof, 2 * self.dof), device=self.device)

        Fx[:, 0, 2] = self.b[:, 0] + 100 * self.coulomb_fric[:, 0] / (
            1 + (100 * x[:, 2]) ** 2
        )
        Fx[:, 1, 3] = self.b[:, 1] + 100 * self.coulomb_fric[:, 1] / (
            1 + (100 * x[:, 3]) ** 2
        )

        return Fx

    def get_Alin(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        A-matrix of the linearized dynamics (xd = Ax+Bu)

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        u : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        torch.Tensor, shape=(num_envs, 4, 4),
            A-matrix

        """
        vel = torch.clone(x[:, self.dof :])

        M = self.mass_matrix(x)
        C = self.coriolis_matrix(x)
        G = self.gravity_vector(x)
        F = self.coulomb_vector(x)

        # Minv = torch.linalg.inv(M)

        Minvx = self.get_Minvx(x, u)
        Cx = self.get_Cx(x, u)
        Gx = self.get_Gx(x, u)
        Fx = self.get_Fx(x, u)

        Alin = torch.zeros(
            (self.num_envs, 2 * self.dof, 2 * self.dof), device=self.device
        )
        Alin[:, 0, 2] = 1.0
        Alin[:, 1, 3] = 1.0

        qddx = torch.zeros((self.num_envs, self.dof, 2 * self.dof), device=self.device)
        qddx[:, 0, 2] = 1.0
        qddx[:, 1, 3] = 1.0

        # lower = torch.dot(
        #     Minvx, (torch.dot(self.B, u) - torch.dot(C, x[:, 2:]) + G - F)
        # ).T + torch.dot(Minv, -torch.dot(Cx, x[:, 2:]).T - torch.dot(C, qddx) + Gx - Fx)
        lower = bmv(
            Minvx, (bmv(self.B, u) - bmv(C, vel) + G - F)
        ).T + torch.linalg.solve(M, -bmv(Cx, vel).T - bmv(C, qddx) + Gx - Fx)
        Alin[:, 2:, :] = lower

        return Alin

    def get_Blin(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        B-matrix of the linearized dynamics (xd = Ax+Bu)

        Parameters
        ----------
        x : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        u : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        torch.Tensor, shape=(num_envs, 4, 2),
            B-matrix

        """
        Blin = torch.zeros((self.num_envs, 2 * self.dof, self.dof), device=self.device)
        M = self.mass_matrix(x)
        # Minv = torch.linalg.inv(M)
        # Blin[:, 2:, :] = torch.dot(Minv, self.B)
        Blin[:, 2:, :] = torch.linalg.solve(M, self.B)

        return Blin

    def linear_matrices(
        self, x0: torch.Tensor, u0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        get A- and B-matrix of the linearized dynamics (xd = Ax+Bu)

        Parameters
        ----------
        x0 : torch.Tensor, shape=(num_envs, 4), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        u0 : torch.Tensor, shape=(num_envs, 2), dtype=float,
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]


        Returns
        -------
        torch.Tensor, shape=(num_envs, 4, 4),
            A-matrix
        torch.Tensor, shape=(num_envs, 4, 2),
            B-matrix

        """
        Alin = self.get_Alin(x0, u0)
        Blin = self.get_Blin(x0, u0)
        return Alin, Blin
