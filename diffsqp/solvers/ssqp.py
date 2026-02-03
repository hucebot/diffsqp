import torch
from torch import bmm

from diffsqp.problems import Problem
from diffsqp.solvers import Admm


class Ssqp:
    def __init__(self, prob: Problem, eps: float = 1e-4) -> None:
        self.eps = eps

        self.prob = prob
        self.horizon = self.prob.horizon

        self.n_batch = self.prob.states[0].shape[0]
        nx = self.prob.n_state
        nu = self.prob.n_ctrl

        self.admm_solver = Admm(prob)

        # Log best line search metrics
        self.best_cost, self.best_gamma = self.calc_metrics(
            self.prob.states,
            self.prob.controls,
        )

        self.terminate = torch.zeros((self.n_batch), dtype=torch.bool)

    def solve(self):
        while not torch.all(self.terminate):
            self.step()

    def step(self):
        delta_x, delta_u = self.admm_solver.solve()
        self.line_search(delta_x, delta_u)
        self.check_termination()

    def line_search(self, delta_x, delta_u, max_iter: float = 10):
        alpha = torch.ones((self.n_batch, 1))
        dones = torch.zeros(self.n_batch, dtype=torch.bool)
        i = 0
        max_iter = 10
        while (not torch.all(dones)) and (i < max_iter):
            print("line_search_iter: ", i)
            i += 1
            gamma = torch.zeros((self.n_batch, 1))

            # Evaluate current alpha
            x_cand, u_cand = self.calc_cadidate_solutions(
                alpha,
                self.prob.states,
                self.prob.controls,
                delta_x,
                delta_u,
            )
            cost, gamma = self.calc_metrics(x_cand, u_cand)

            # Update successful batches
            cost_improved = cost < self.best_cost
            gamma_improved = gamma < self.best_gamma
            update_mask = (cost_improved | gamma_improved) & ~dones
            if update_mask.any():
                for k in range(self.horizon - 1):
                    self.prob.states[k][update_mask] = x_cand[k][update_mask]
                    self.prob.controls[k][update_mask] = u_cand[k][update_mask]
                self.prob.states[-1][update_mask] = x_cand[-1][update_mask]
                # Mark batches as finished
                dones[update_mask] = True

            # Update best cost and gamma
            self.best_cost[cost_improved & ~dones] = cost[cost_improved & ~dones]
            self.best_gamma[gamma_improved & ~dones] = gamma[gamma_improved & ~dones]

            # Update alpha for failed batches
            failed_mask = ~update_mask & ~dones
            if failed_mask.any():
                alpha[failed_mask] *= 0.5

    def check_termination(self):
        for k in range(self.horizon - 1):
            viol = self.calc_dynamics_violation(
                self.prob.stage_dynamics[k].f,
                self.prob.states[k + 1],
                self.prob.states[k],
                self.prob.controls[k],
            )
            self.terminate = torch.norm(viol, p=float("inf"), dim=1) < self.eps

    def calc_cadidate_solutions(self, alpha, curr_x, curr_u, delta_x, delta_u):
        x_cand = []
        u_cand = []
        for k in range(self.horizon):
            x_cand.append(curr_x[k] + torch.mul(alpha, delta_x[k]))
            if k < self.horizon - 1:
                u_cand.append(curr_u[k] + torch.mul(alpha, delta_u[k]))
        return x_cand, u_cand

    def calc_metrics(self, x_cand, u_cand):
        cost = torch.zeros((self.n_batch))
        gamma = torch.zeros((self.n_batch))
        for k in range(self.horizon - 1):
            # Calculate total trajectory cost
            cost += self.prob.costs[k].l(x_cand[k], u_cand[k])
            # Calculate constraint violations
            viol = self.calc_dynamics_violation(
                self.prob.stage_dynamics[k].f,
                x_cand[k + 1],
                x_cand[k],
                u_cand[k],
            )
            gamma += torch.norm(viol, p=float("inf"), dim=1)
        # Add final node cost
        cost += self.prob.costs[-1].l(x_cand[-1])
        return cost, gamma

    def calc_dynamics_violation(self, f, x_next, x, u):
        return x_next - f(x, u, self.prob.dt)
