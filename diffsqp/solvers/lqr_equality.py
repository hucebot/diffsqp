import torch

from diffsqp.problems import Problem
from diffsqp.utils.math import mm, mv


class EqualityConstrainedLqr:
    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.horizon = self.prob.horizon
        self.n_batch = self.prob.states[0].shape[0]
        self.n_state = self.prob.n_state
        self.n_ctrl = self.prob.n_ctrl

        self.K = [None] * (self.horizon - 1)
        self.k = [None] * (self.horizon - 1)

        self.V = [None] * self.horizon
        self.v = [None] * self.horizon

        self.Dx = [None] * self.horizon
        self.Du = [None] * (self.horizon - 1)
        self.Dpi = [None] * (self.horizon - 1)
        self.Dlam = [None] * (self.horizon - 1)

        # Only for API cohesion, does nothing (for now)
        self.delta_lambda = [torch.zeros((self.n_batch, self.n_state))] * (self.horizon)

    def solve(self):
        self.backward_pass_()
        self.forward_pass_()

        # Return corrections
        return self.Dx, self.Du, self.Dlam

    def backward_pass_(self):
        x_N = self.prob.states[self.horizon - 1]
        self.V[-1], self.v[-1] = self.calc_final_cost_terms_(x_N, self.prob.costs[-1])
        for i in range(self.horizon - 2, -1, -1):
            x_lin = self.prob.states[i]
            u_lin = self.prob.controls[i]
            x_next = self.prob.states[i + 1]

            Q, R, S, q, r = self.calc_linearized_cost_terms_(
                x_lin, u_lin, self.prob.costs[i]
            )
            A, B, b = self.calc_linearized_dynamic_terms_(
                x_lin, u_lin, x_next, self.prob.stage_dynamics[i]
            )
            C, D, e = self.calc_linearized_constraint_terms_(
                x_lin, u_lin, self.prob.stage_dynamics[i]
            )

            (self.K[i], self.k[i], self.V[i], self.v[i]) = self.riccati_backward_(
                Q=Q,
                q=q,
                R=R,
                r=r,
                S=S,
                V=self.V[i + 1],
                v=self.v[i + 1],
                A=A,
                B=B,
                b=b,
                C=C,
                D=D,
                e=e,
            )

    def forward_pass_(self):
        # TODO: Add initial state optimization as an option
        nx = self.prob.n_state
        self.Dx[0] = torch.zeros([self.n_batch, nx])
        for i in range(self.horizon - 1):
            x_lin = self.prob.states[i]
            u_lin = self.prob.controls[i]
            x_next = self.prob.states[i + 1]

            Dx0 = self.Dx[i]
            K = self.K[i]
            k = self.k[i]
            A, B, b = self.calc_linearized_dynamic_terms_(
                x_lin, u_lin, x_next, self.prob.stage_dynamics[i]
            )
            self.Dx[i + 1], self.Du[i], self.Dlam[i] = self.riccati_forward_(
                Dx0=Dx0, K=K, k=k, A=A, B=B, b=b
            )

    def riccati_backward_(self, Q, q, R, r, S, V, v, A, B, b, C, D, e):
        nB = self.n_batch
        nx = self.prob.n_state
        nu = self.prob.n_ctrl
        ng = D.shape[1]  # n of equality constraints
        assert ng == self.prob.stage_dynamics[0].ng

        AT = torch.transpose(A, 1, 2)
        BT = torch.transpose(B, 1, 2)
        ST = torch.transpose(S, 1, 2)
        DT = torch.transpose(D, 1, 2)

        Q_ = Q + mm(AT, mm(V, A))
        l = mv(V, b) + v
        q_ = q + mv(AT, l)
        R_ = R + mm(BT, mm(V, B))
        r_ = r + mv(BT, l)
        S_ = S + mm(BT, mm(V, A))

        S_T = S_.transpose(1, 2)

        S_ext = torch.cat([S_, C], dim=1)
        r_ext = torch.cat([-r_, e], dim=1)
        S_extT = S_ext.transpose(1, 2)

        dim = R_.shape[1] + D.shape[1]
        R_ext = torch.zeros((nB, dim, dim))
        R_ext[:, 0:nu, 0:nu] = R_
        R_ext[:, nu:, 0:nu] = D
        R_ext[:, 0:nu, nu:] = DT

        K_ext = torch.linalg.solve(R_ext, -S_ext)
        k_ext = torch.linalg.solve(R_ext, -r_ext)

        V_ = Q_ + mm(S_extT, K_ext)
        v_ = q_ - mv(S_extT, -k_ext)

        # K_ = K_ext[:, 0:nu, :]
        # k_ = k_ext[:, 0:nu]

        # Sanity checks
        nB = self.n_batch
        nx = self.prob.n_state
        nu = self.prob.n_ctrl
        assert Q_.shape == torch.Size([nB, nx, nx])
        assert q_.shape == torch.Size([nB, nx])
        assert R_.shape == torch.Size([nB, nu, nu])
        assert r_.shape == torch.Size([nB, nu])
        assert K_ext.shape == torch.Size([nB, nu + ng, nx])
        assert k_ext.shape == torch.Size([nB, nu + ng])
        assert V_.shape == torch.Size([nB, nx, nx])
        assert v_.shape == torch.Size([nB, nx])

        return K_ext, k_ext, V_, v_

    def riccati_forward_(self, Dx0, K, k, A, B, b):
        nu = self.prob.n_ctrl
        Du = mv(K[:, 0:nu, :], Dx0) + k[:, 0:nu]
        Dx = mv(A, Dx0) + mv(B, Du) + b
        Dlam = mv(K[:, nu:, :], Dx0) + k[:, nu:]

        # Sanity checks
        nB = self.n_batch
        nx = self.prob.n_state
        ng = K.shape[1] - nu
        assert Dx.shape == torch.Size([nB, nx])
        assert Du.shape == torch.Size([nB, nu])
        assert Dlam.shape == torch.Size([nB, ng])

        return Dx, Du, Dlam

    def calc_linearized_cost_terms_(self, x_lin, u_lin, c):
        Q = c.lxx(x_lin, u_lin)
        q = c.lx(x_lin, u_lin)
        R = c.luu(x_lin, u_lin)
        r = c.lu(x_lin, u_lin)
        S = c.lux(x_lin, u_lin)
        return Q, R, S, q, r

    def calc_linearized_dynamic_terms_(self, x_lin, u_lin, x_next, dyn):
        x_pred = dyn.f(x_lin, u_lin, self.prob.dt)
        b = x_pred - x_next
        A = dyn.fx(x_lin, u_lin, self.prob.dt)
        B = dyn.fu(x_lin, u_lin, self.prob.dt)
        return A, B, b

    def calc_final_cost_terms_(self, x_N, final_cost):
        V = final_cost.lxx(x_N)
        v = final_cost.lx(x_N)
        return V, v

    def calc_linearized_constraint_terms_(self, x_lin, u_lin, dyn):
        C = dyn.gx(x_lin, u_lin)
        D = dyn.gu(x_lin, u_lin)
        e = dyn.g(x_lin, u_lin)
        return C, D, e
