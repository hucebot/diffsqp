import torch
from torch import sin, cos

from diffsqp.dynamics import Dynamics


class CartPoleInverseDynamicsConstrained(Dynamics):
    def __init__(
        self,
        mc: float,
        mp: float,
        lp: float,
        constr_u: bool = False,
        constr_value: float = 0.0,
        grav: float = 9.81,
    ):
        self.type = "inverse"
        self.nx = 4
        self.nq = 2
        self.nv = 2
        self.nu = 2
        self.ng = 2

        self.constr_u = constr_u
        self.constr_value = constr_value

        self.mc = mc
        self.mp = mp
        self.lp = lp
        self.grav = grav

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """

        ds = x[:, 2:3]
        dth = x[:, 3:4]
        dds = u[:, 0:1]
        ddth = u[:, 1:2]

        return torch.cat([ds, dth, dds, ddth], dim=1)

    def fcx(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        n_batch = x.shape[0]
        grad = torch.zeros(self.nx, self.nx).repeat(n_batch, 1, 1)
        grad[:, 0 : self.nv, self.nv :] = torch.eye(self.nv)
        return grad

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        n_batch = x.shape[0]
        grad = torch.zeros(self.nx, self.nu).repeat(n_batch, 1, 1)
        grad[:, self.nv :, :] = torch.eye(self.nu)
        return grad

    def g(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        mc = self.mc
        mp = self.mp
        lp = self.lp
        grav = self.grav

        th = x[:, 1:2]
        dds = u[:, 0:1]
        ddth = u[:, 1:2]

        sth = torch.sin(th)
        cth = torch.cos(th)

        res = torch.zeros((n_batch, self.ng))
        res[:, 0:1] = mp * lp * cth * dds + mp * lp * lp * ddth + mp * grav * lp * sth
        if self.constr_u:
            res[:, 1:2] = dds - self.constr_value
        return res

    def gx(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        ng = self.ng
        nx = self.nx

        mp = self.mp
        lp = self.lp
        grav = self.grav
        th = x[:, 1:2]
        dds = u[:, 0:1]

        sth = torch.sin(th)
        cth = torch.cos(th)

        grad = torch.zeros((n_batch, ng, nx))
        grad[:, 0:1, 1:2] = (-mp * lp * sth * dds + mp * grav * lp * cth).unsqueeze(2)
        return grad

    def gu(self, x: torch.Tensor, u: torch.Tensor):
        n_batch = x.shape[0]
        ng = self.ng
        nu = self.nu

        mp = self.mp
        lp = self.lp
        grav = self.grav
        th = x[:, 1:2]
        sth = torch.sin(th)
        cth = torch.cos(th)

        grad = torch.zeros((n_batch, ng, nu))
        grad[:, 0:1, 0:1] = (mp * lp * cth).unsqueeze(2)
        grad[:, 0:1, 1:] = mp * lp * lp
        if self.constr_u:
            grad[:, 1:2, 0] = 1.0
        return grad
