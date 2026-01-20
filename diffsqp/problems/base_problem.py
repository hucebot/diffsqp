import torch
from abc import ABC, abstractmethod

from diffsqp.costs import Cost
from diffsqp.dynamics import Dynamics


class Problem(ABC):
    def __init__(self, horizon: int, dt: float, n_state: int, n_ctrl: int) -> None:
        self.horizon = horizon
        self.dt = dt
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.costs: list[Cost] = []
        self.stage_dynamics: list[Dynamics] = []
        self.variables: torch.Tensor | None = None
        self.multipliers: torch.Tensor | None = None

    def state(self, stage_idx: int) -> torch.Tensor:
        # TODO: Add check here to see if index corresponds to horizon length
        start = stage_idx * (self.n_state + self.n_ctrl)
        end = start + self.n_state
        return self.variables[:, start:end]

    def control(self, i: int) -> torch.Tensor:
        # TODO: Add check here to see if index corresponds to horizon length
        start = i * (self.n_state + self.n_ctrl) + self.n_state
        end = start + self.n_ctrl
        return self.variables[:, start:end]

    def set_state(self, i: int, val: torch.Tensor) -> None:
        start = i * (self.n_state + self.n_ctrl)
        end = start + self.n_state
        self.variables[:, start:end] = val

    def set_control(self, i: int, val: torch.Tensor) -> None:
        start = i * (self.n_state + self.n_ctrl) + self.n_state
        end = start + self.n_ctrl
        self.variables[:, start:end] = val
