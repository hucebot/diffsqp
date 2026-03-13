import torch

from diffsqp.costs import Cost


class LqrCost(Cost):
    def __init__(self, Q, R, x_des=None, u_des=None):
        self.nB = Q.shape[0]
        self.nx = Q.shape[1]
        self.nu = R.shape[1]
        self.Q = Q
        self.R = R

        if x_des is None:
            x_des = torch.zeros((self.nB, self.nx))
        if u_des is None:
            u_des = torch.zeros((self.nB, self.nu))
        self.x_des = x_des
        self.u_des = u_des

    def l(self, x, u):
        x_term = torch.bmm(
            torch.bmm(torch.transpose((x - self.x_des).unsqueeze(2), 1, 2), self.Q),
            (x - self.x_des).unsqueeze(2),
        ).squeeze(1, 2)
        u_term = torch.bmm(
            torch.bmm(torch.transpose((u - self.u_des).unsqueeze(2), 1, 2), self.R),
            (u - self.u_des).unsqueeze(2),
        ).squeeze(1, 2)
        return 0.5 * (x_term + u_term)

    def lx(self, x, u):
        """Gradient w.r.t x (B, nx, 1)"""
        return torch.bmm(self.Q, (x - self.x_des).unsqueeze(2)).squeeze(2)

    def lu(self, x, u):
        """Gradient w.r.t u (B, nu, 1)"""
        return torch.bmm(self.R, (u - self.u_des).unsqueeze(2)).squeeze(2)

    def lxx(self, x, u):
        """Hessian w.r.t xx (B, nx, nx)"""
        return self.Q

    def luu(self, x, u):
        """Hessian w.r.t uu (B, nu, nu)"""
        return self.R

    def lux(self, x, u):
        """Hessian w.r.t ux (B, nu, nx)"""
        return torch.zeros(self.nB, self.nu, self.nx, device=x.device, dtype=x.dtype)

    def lxu(self, x, u):
        """Hessian w.r.t xu (B, nx, nu)"""
        return torch.zeros(self.nB, self.nx, self.nu, device=x.device, dtype=x.dtype)
