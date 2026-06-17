import torch

from diffsqp.dynamics import Dynamics


class PendulumDynamics(Dynamics):
    def __init__(self, m: float, l: float, b: float, grav: float = 9.81):
        self.type = "forward"
        self.nx = 2
        self.nq = 1
        self.nv = 1
        self.nu = 1
        self.m = m
        self.l = l
        self.b = b
        self.grav = grav

        # Pre-calculate constant factor
        self.inertia_inv = 1.0 / (self.m * self.l**2)

    def fc(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Continuous time dynamics: x_dot = fc(x_k, u_k)
        """
        theta = x[..., 0:1]
        dtheta = x[..., 1:2]

        ddtheta = (
            -(self.grav / self.l) * torch.sin(theta)
            - (self.b * self.inertia_inv) * dtheta
            + (self.inertia_inv) * u
        )

        return torch.cat([dtheta, ddtheta], dim=-1)

    def fcx(self, x, u) -> torch.Tensor:
        """
        dfc/dx matrix:
        [ 0.0,               1.0     ]
        [ -g/l * cos(theta), -b/ml^2 ]
        """
        n_B = x.shape[:-1]

        # Initialize Jacobian tensor (Batch, State, State)
        A = torch.zeros((*n_B, self.nx, self.nx))
        A[..., 0, 1] = 1.0
        A[..., 1, 0] = -(self.grav / self.l) * torch.cos(x[:, 0])
        A[..., 1, 1] = -self.b * self.inertia_inv
        return A

    def fcu(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        dfc/du matrix:
        [ 0.0    ]
        [ 1/ml^2 ]
        """
        n_B = x.shape[:-1]
        B = torch.zeros((*n_B, self.nx, self.nu))
        B[:, 1, 0] = self.inertia_inv
        return B
