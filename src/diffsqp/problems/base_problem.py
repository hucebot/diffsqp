import torch
from abc import ABC, abstractmethod

from diffsqp.costs import Cost, TerminalCost
from diffsqp.dynamics import Dynamics
from diffsqp.constraints import Constraint


class Problem(ABC):
    def __init__(self, horizon: int, dt: float, nx: int, nu: int) -> None:
        self.horizon = horizon
        self.dt = dt
        self.nx = nx
        self.nu = nu
        self.costs: List[List[Cost | TerminalCost]] = []
        self.stage_dynamics: List[Dynamics] = []
        self.constraints: List[Constraints] = [None] * self.horizon
        self.states: List[torch.Tensor] = []
        self.controls: List[torch.Tensor] = []
        self.costates: List[torch.Tensor] = []

    def l(self, stage_idx, x, u=None):
        all_costs = torch.stack(
            [c.l(x, u) if u is not None else c.l(x) for c in self.costs[stage_idx]]
        )
        return torch.sum(all_costs, dim=0)

    def lx(self, stage_idx, x, u=None):
        all_grads = torch.stack(
            [c.lx(x, u) if u is not None else c.lx(x) for c in self.costs[stage_idx]]
        )
        return torch.sum(all_grads, dim=0)

    def lu(self, stage_idx, x, u):
        all_grads = torch.stack(
            [c.lu(x, u) if u is not None else c.lu(x) for c in self.costs[stage_idx]]
        )
        return torch.sum(all_grads, dim=0)

    def lxx(self, stage_idx, x, u=None):
        all_hessians = torch.stack(
            [c.lxx(x, u) if u is not None else c.lxx(x) for c in self.costs[stage_idx]]
        )
        return torch.sum(all_hessians, dim=0)

    def luu(self, stage_idx, x, u):
        all_hessians = torch.stack(
            [c.luu(x, u) if u is not None else c.luu(x) for c in self.costs[stage_idx]]
        )
        return torch.sum(all_hessians, dim=0)

    def lux(self, stage_idx, x, u):
        all_hessians = torch.stack(
            [c.lux(x, u) if u is not None else c.lux(x) for c in self.costs[stage_idx]]
        )
        return torch.sum(all_hessians, dim=0)

    def lxu(self, stage_idx, x, u):
        all_hessians = torch.stack(
            [c.lxu(x, u) if u is not None else c.lxu(x) for c in self.costs[stage_idx]]
        )
        return torch.sum(all_hessians, dim=0)

    def g(self, stage_idx, x, u=None):
        constr = torch.cat([c.g(x, u) for c in self.constraints[stage_idx]], dim=1)
        return constr

    def gx(self, stage_idx, x, u):
        grad = torch.cat([c.gx(x, u) for c in self.constraints[stage_idx]], dim=1)
        return grad

    def gu(self, stage_idx, x, u):
        grad = torch.cat([c.gu(x, u) for c in self.constraints[stage_idx]], dim=1)
        return grad

    # def state(self, stage_idx: int) -> torch.Tensor:
    #     # TODO: Add check here to see if index corresponds to horizon length
    #     start = stage_idx * (self.nx + self.nu)
    #     end = start + self.nx
    #     return self.variables[:, start:end]
    #
    # def control(self, i: int) -> torch.Tensor:
    #     # TODO: Add check here to see if index corresponds to horizon length
    #     start = i * (self.nx + self.nu) + self.nx
    #     end = start + self.nu
    #     return self.variables[:, start:end]
    #
    # def set_state(self, i: int, val: torch.Tensor) -> None:
    #     start = i * (self.nx + self.nu)
    #     end = start + self.nx
    #     self.variables[:, start:end] = val
    #
    # def set_control(self, i: int, val: torch.Tensor) -> None:
    #     start = i * (self.nx + self.nu) + self.nx
    #     end = start + self.nu
    #     self.variables[:, start:end] = val
