import torch
from torch import sin, cos

from dataclasses import dataclass

from diffsqp.dynamics import Dynamics


class AcrobotParameters:
    def __init__(self, **args):
        self.name: str = args["name"]

        self.n_x: int = args["n_x"]
        self.n_q: int = args["n_q"]
        self.n_v: int = args["n_v"]
        self.n_j: int = args["n_j"]
        self.n_u: int = args["n_u"]

        self.m1: float = args["m1"]
        self.m2: float = args["m2"]
        self.l1: float = args["l1"]
        self.l2: float = args["l2"]
        self.lc1: float = args["lc1"]
        self.lc2: float = args["lc2"]
        self.I1: float = (
            args["I1"] if args["I1"] is not None else 1 / (3.0 * self.m1 * self.l1**2)
        )
        self.I2: float = (
            args["I2"] if args["I2"] is not None else 1 / (3.0 * self.m2 * self.l2**2)
        )

        self.grav: float = args["grav"]


class AcrobotDynamics(Dynamics):
    def __init__(self, params: AcrobotParameters):
        self.p = params
        super().__init__(nx=self.p.n_x, nu=self.p.n_u, nq=self.p.n_q, nv=self.p.n_v)

        if self.p.I1 is None:
            self.p.I1 = (self.p.m1 * self.p.l1**2) / 3.0
        if self.p.I2 is None:
            self.p.I2 = (self.p.m2 * self.p.l2**2) / 3.0

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """
        m1 = self.p.m1
        m2 = self.p.m2
        l1 = self.p.l1
        l2 = self.p.l2
        lc1 = self.p.lc1
        lc2 = self.p.lc2
        grav = self.p.grav
        I1 = self.p.I1
        I2 = self.p.I2

        th1 = x[..., 0:1]
        th2 = x[..., 1:2]
        dth1 = x[..., 2:3]
        dth2 = x[..., 3:4]
        tau = u[..., 0:1]

        s1 = sin(th1)
        s2 = sin(th2)
        c2 = cos(th2)
        s12 = sin(th1 + th2)

        denom = I1 * I2 + I2 * l1**2 * m2 - l1**2 * lc2**2 * m2**2 * c2**2
        nom1 = -I2 * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * s2 * dth2
        )
        nom2 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth1**2 - tau
        )
        ddth1 = (nom1 + nom2) / denom

        # denom = I1 * I2 + I2 * l1**2 * m2 - l1**2 * lc2**2 * m2**2 * c2**2
        nom1 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * s2 * dth2
        )
        nom2 = -(grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth2 * dth2 - tau) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )
        ddth2 = (nom1 + nom2) / denom

        return torch.cat([dth1, dth2, ddth1, ddth2], dim=-1)

    def fcx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/dx matrix: n_x x n_x
        """
        n_B = x.shape[:-1]

        m1 = self.p.m1
        m2 = self.p.m2
        l1 = self.p.l1
        l2 = self.p.l2
        lc1 = self.p.lc1
        lc2 = self.p.lc2
        grav = self.p.grav
        I1 = self.p.I1
        I2 = self.p.I2

        th1 = x[..., 0:1]
        th2 = x[..., 1:2]
        dth1 = x[..., 2:3]
        dth2 = x[..., 3:4]
        tau = u[..., 0:1]

        s1 = sin(th1)
        s2 = sin(th2)
        c2 = cos(th2)
        s12 = sin(th1 + th2)
        c1 = cos(th1)
        c12 = cos(th1 + th2)

        d_nom1_th1 = -I2 * (
            grav * l1 * m2 * c1 + grav * lc1 * m1 * c1 + grav * lc2 * m2 * c12
        )
        d_nom2_th1 = (I2 + l1 * lc2 * m2 * c2) * (grav * lc2 * m2 * c12)
        denom = I1 * I2 + I2 * l1 * l1 * m2 - l1 * l1 * lc2 * lc2 * m2 * m2 * c2 * c2

        # DDTH1_TH1
        d_ddth1_th1 = (d_nom1_th1 + d_nom2_th1) / denom

        d_denom_th2 = 2 * l1 * l1 * lc2 * lc2 * m2 * m2 * c2 * s2
        d_nom1_th2 = -I2 * (
            grav * lc2 * m2 * c12 - l1 * lc2 * m2 * (2 * dth1 + dth2) * c2 * dth2
        )
        A = I2 + l1 * lc2 * m2 * c2
        B = grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth1 * dth1 - tau
        dA_th2 = -l1 * lc2 * m2 * s2
        dB_th2 = grav * lc2 * m2 * c12 + l1 * lc2 * m2 * c2 * dth1 * dth1
        d_nom2_th2 = dA_th2 * B + A * dB_th2
        nom1 = -I2 * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2 * dth1 + dth2) * s2 * dth2
        )
        nom2 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth1 * dth1 - tau
        )
        nom_total = nom1 + nom2
        d_nom_total_th2 = d_nom1_th2 + d_nom2_th2

        # DDTH1_TH2
        d_ddth1_th2 = (d_nom_total_th2 * denom - nom_total * d_denom_th2) / (
            denom * denom
        )

        d_nom1_dth1 = -I2 * (-l1 * lc2 * m2 * 2 * s2 * dth2)
        d_nom2_dth1 = (I2 + l1 * lc2 * m2 * c2) * (l1 * lc2 * m2 * s2 * 2 * dth1)

        # DDTH1_DTH1
        d_ddth1_dth1 = (d_nom1_dth1 + d_nom2_dth1) / denom

        d_nom1_dth2 = -I2 * (-l1 * lc2 * m2 * (2 * dth1 + 2 * dth2) * s2)
        d_nom2_dth2 = 0

        # DDTH1_DTH2
        d_ddth1_dth2 = (d_nom1_dth2 + d_nom2_dth2) / denom

        d_nom1_th1 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * l1 * m2 * c1 + grav * lc1 * m1 * c1 + grav * lc2 * m2 * c12
        )
        d_nom2_th1 = -(grav * lc2 * m2 * c12) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )

        # DDTH2_TH1
        d_ddth2_th1 = (d_nom1_th1 + d_nom2_th1) / denom

        d_denom_th2 = 2.0 * l1 * l1 * lc2 * lc2 * m2 * m2 * c2 * s2
        A1 = I2 + l1 * lc2 * m2 * c2
        B1 = (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * s2 * dth2
        )
        dA1_th2 = -l1 * lc2 * m2 * s2
        dB1_th2 = (
            grav * lc2 * m2 * c12 - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * c2 * dth2
        )
        d_nom1_th2 = dA1_th2 * B1 + A1 * dB1_th2
        A2 = -(grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth2 * dth2 - tau)
        B2 = I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        dA2_th2 = -(grav * lc2 * m2 * c12 + l1 * lc2 * m2 * c2 * dth2 * dth2)
        dB2_th2 = -2.0 * l1 * lc2 * m2 * s2
        d_nom2_th2 = dA2_th2 * B2 + A2 * dB2_th2
        nom1 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * dth1 + dth2) * s2 * dth2
        )
        nom2 = -(grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * dth2 * dth2 - tau) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )
        nom_total_2 = nom1 + nom2
        d_nom_total_2_th2 = d_nom1_th2 + d_nom2_th2

        # DDTH2_TH2
        d_ddth2_th2 = (d_nom_total_2_th2 * denom - nom_total_2 * d_denom_th2) / (
            denom * denom
        )

        d_nom1_dth1 = (I2 + l1 * lc2 * m2 * c2) * (-l1 * lc2 * m2 * 2.0 * s2 * dth2)
        d_nom2_dth1 = 0.0

        # DDTH2_DTH2
        d_ddth2_dth1 = (d_nom1_dth1 + d_nom2_dth1) / denom

        d_nom1_dth2 = (I2 + l1 * lc2 * m2 * c2) * (
            -l1 * lc2 * m2 * s2 * (2.0 * dth1 + 2.0 * dth2)
        )
        d_nom2_dth2 = -(l1 * lc2 * m2 * s2 * 2.0 * dth2) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )

        # DDTH2_DTH2
        d_ddth2_dth2 = (d_nom1_dth2 + d_nom2_dth2) / denom

        A = torch.zeros((*n_B, self.nx, self.nx), device=x.device)
        A[..., 0, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        A[..., 1, :] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        A[..., 2, 0] = d_ddth1_th1.squeeze(1)
        A[..., 2, 1] = d_ddth1_th2.squeeze(1)
        A[..., 2, 2] = d_ddth1_dth1.squeeze(1)
        A[..., 2, 3] = d_ddth1_dth2.squeeze(1)
        A[..., 3, 0] = d_ddth2_th1.squeeze(1)
        A[..., 3, 1] = d_ddth2_th2.squeeze(1)
        A[..., 3, 2] = d_ddth2_dth1.squeeze(1)
        A[..., 3, 3] = d_ddth2_dth2.squeeze(1)

        return A

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/du matrix: n_x x n_u
        """
        n_B = x.shape[:-1]

        m2 = self.p.m2
        l1 = self.p.l1
        lc2 = self.p.lc2
        I1 = self.p.I1
        I2 = self.p.I2

        th2 = x[..., 1:2]
        tau = u[..., 0:1]

        c2 = cos(th2)

        denom = I1 * I2 + I2 * l1 * l1 * m2 - l1 * l1 * lc2 * lc2 * m2 * m2 * c2 * c2
        d_nom2_1_tau = -(I2 + l1 * lc2 * m2 * c2)
        d_ddth1_tau = d_nom2_1_tau / denom

        d_nom2_2_tau = I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        d_ddth2_tau = d_nom2_2_tau / denom

        B = torch.zeros((*n_B, self.nx, self.nu), device=x.device)
        B[..., 2, 0] = d_ddth1_tau[..., 0]
        B[..., 3, 0] = d_ddth2_tau[..., 0]

        return B
