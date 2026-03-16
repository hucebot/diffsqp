from abc import ABC, abstractmethod
from typing import List, Optional
import torch

from diffsqp.costs import Cost
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import Constraint


class Problem(ABC):
    """
    An abstract base class representing a Trajectory Optimization problem.

    Attributes:
        horizon (int): The total number of time steps (T).
        dt (float): The integration time step.
        nx (int): Dimension of the state vector.
        nu (int): Dimension of the control vector.
        costs (List[List[Cost]]): A list of length `horizon`, where each element
            is a list of Cost objects active at that stage.
        dynamics (List[Dynamics]): A list of dynamics models for each transition.
        constraints (List[Constraints]): A list of constraint objects for each stage.
        states (List[torch.Tensor]): The current state trajectory [nB x nx].
        controls (List[torch.Tensor]): The current control trajectory [nB x nu].
        pi (List[torch.Tensor]): Lagrange multipliers for dynamics (equality) constraints.
        ni (List[torch.Tensor]): Lagrange multipliers for general equality constraints.
    """

    def __init__(self, horizon: int, dt: float, nB: int, nx: int, nu: int) -> None:
        """
        Initializes the optimization problem buffers.

        Args:
            horizon (int): Horizon length.
            dt (float): Integration dt.
            nB (int): Batch size for parallel trajectory optimization.
            nx (int): State dimension.
            nu (int): Control dimension.
        """
        self.horizon = horizon
        self.dt = dt
        self.nx = nx
        self.nu = nu
        self.costs: List[List[Cost]] = []
        self.dynamics: List[Dynamics] = []
        self.constraints: List[Constraints] = [None] * self.horizon
        self.states: List[torch.Tensor] = [
            torch.zeros((nB, nx)) for _ in range(self.horizon)
        ]
        self.controls: List[torch.Tensor] = [
            torch.zeros((nB, nu)) for _ in range(self.horizon - 1)
        ]
        self.pi: List[torch.Tensor] = [
            torch.zeros((nB, nx)) for _ in range(self.horizon - 1)
        ]
        self.ni: List[torch.Tensor] = [None for _ in range(self.horizon)]

    # --- Cost Aggregation Methods ---

    def l(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the total stage cost by summing all cost components.

        Args:
            stage_idx: The current time step index.
            x: State tensor [nB x nx].
            u: Control tensor [nB x nu]. Optional for terminal stage.

        Returns:
            Total scalar cost per batch [nB].
        """
        all_costs = torch.stack([c.l(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_costs, dim=0)

    def lx(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Total state gradient of the cost at stage_idx."""
        all_grads = torch.stack([c.lx(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_grads, dim=0)

    def lu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total control gradient of the cost at stage_idx."""
        all_grads = torch.stack([c.lu(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_grads, dim=0)

    def lxx(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Total state Hessian (d^2L/dx^2) at stage_idx."""
        all_hessians = torch.stack([c.lxx(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_hessians, dim=0)

    def luu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total control Hessian (d^2L/du^2) at stage_idx."""
        all_hessians = torch.stack([c.luu(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_hessians, dim=0)

    def lux(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total cross-derivative (d^2L/dudx) at stage_idx."""
        all_hessians = torch.stack([c.lux(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_hessians, dim=0)

    def lxu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Total cross-derivative (d^2L/dxdu) at stage_idx."""
        all_hessians = torch.stack([c.lxu(x, u) for c in self.costs[stage_idx]])
        return torch.sum(all_hessians, dim=0)

    # --- Constraint Aggregation Methods ---

    def g(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Concatenate all stage constraints into a single vector.

        Returns:
            A tensor of concatenated constraints [nB x total_constraints].
        """
        constr = torch.cat([c.g(x, u) for c in self.constraints[stage_idx]], dim=1)
        return constr

    def gx(
        self, stage_idx: int, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Jacobian of the aggregated constraints with respect to state x."""
        grad = torch.cat([c.gx(x, u) for c in self.constraints[stage_idx]], dim=1)
        return grad

    def gu(self, stage_idx: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Jacobian of the aggregated constraints with respect to control u."""
        grad = torch.cat([c.gu(x, u) for c in self.constraints[stage_idx]], dim=1)
        return grad
