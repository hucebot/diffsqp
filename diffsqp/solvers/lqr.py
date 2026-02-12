import torch

from diffsqp.problems import Problem


def mm(A, B):
    return torch.bmm(A, B)


def mv(A, b):
    return torch.bmm(A, b.unsqueeze(2)).squeeze(2)


class Lqr:
    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.horizon = self.prob.horizon
        self.n_batch = self.prob.states[0].shape[0]
        self.n_state = self.prob.n_state
        self.n_ctrl = self.prob.n_ctrl

        self.Q_bar = [None] * (self.horizon - 1)
        self.q_bar = [None] * (self.horizon - 1)
        self.R_bar = [None] * (self.horizon - 1)
        self.r_bar = [None] * (self.horizon - 1)
        self.S_bar = [None] * (self.horizon - 1)

        self.K = [None] * (self.horizon - 1)
        self.k = [None] * (self.horizon - 1)

        self.V = [None] * self.horizon
        self.v = [None] * self.horizon

        self.Dx = [None] * self.horizon
        self.Du = [None] * (self.horizon - 1)

        # Only for API cohesion, does nothing (for now)
        self.delta_lambda = [torch.zeros((self.n_batch, self.n_state))] * (self.horizon)

    def solve(self):
        self.backward_pass()
        self.forward_pass()

        # Return corrections
        return self.Dx, self.Du

    def backward_pass(self):
        x_N = self.prob.states[self.horizon - 1]
        self.V[-1] = self.prob.costs[-1].lxx(x_N)
        self.v[-1] = self.prob.costs[-1].lx(x_N)

        for i in range(self.horizon - 2, -1, -1):
            x = self.prob.states[i]
            u = self.prob.controls[i]
            x_next = self.prob.stage_dynamics[i].f(x, u, self.prob.dt)
            b = x_next - self.prob.states[i + 1]

            Q = self.prob.costs[i].lxx(x, u)
            q = self.prob.costs[i].lx(x, u)
            R = self.prob.costs[i].luu(x, u)
            r = self.prob.costs[i].lu(x, u)

            S = self.prob.costs[i].lux(x, u)

            A = self.prob.stage_dynamics[i].fx(x, u, self.prob.dt)
            B = self.prob.stage_dynamics[i].fu(x, u, self.prob.dt)

            (
                self.V[i],
                self.v[i],
                self.K[i],
                self.k[i],
                self.Q_bar[i],
                self.q_bar[i],
                self.R_bar[i],
                self.r_bar[i],
            ) = self.riccati_backward_(
                Q=Q, q=q, R=R, r=r, S=S, A=A, B=B, b=b, V=self.V[i + 1], v=self.v[i + 1]
            )

    def forward_pass(self):
        nB = self.n_batch
        nx = self.prob.n_state

        self.Dx[0] = torch.zeros([nB, nx])
        for i in range(self.horizon - 1):

            Dx0 = self.Dx[i]
            K = self.K[i]
            k = self.k[i]

            x = self.prob.states[i]
            u = self.prob.controls[i]
            A = self.prob.stage_dynamics[i].fx(x, u, self.prob.dt)
            B = self.prob.stage_dynamics[i].fu(x, u, self.prob.dt)
            x_next = self.prob.stage_dynamics[i].f(x, u, self.prob.dt)
            b = x_next - self.prob.states[i + 1]

            self.Dx[i + 1], self.Du[i] = self.riccati_forward_(
                Dx0=Dx0, K=K, k=k, A=A, B=B, b=b
            )

    def riccati_backward_(self, Q, q, R, r, S, A, B, b, V, v):
        AT = torch.transpose(A, 1, 2)
        BT = torch.transpose(B, 1, 2)
        ST = torch.transpose(S, 1, 2)

        Q_ = Q + mm(AT, mm(V, A))
        l = mv(V, b) + v
        q_ = q + mv(AT, l)
        R_ = R + mm(BT, mm(V, B))
        r_ = r + mv(BT, l)
        S_ = S + mm(BT, mm(V, A))

        S_T = S_.transpose(1, 2)
        K_ = torch.linalg.solve(R_, -S_)
        k_ = torch.linalg.solve(R_, -r_)

        V_ = Q_ - mm(S_T, -K_)
        v_ = q_ - mv(S_T, -k_)

        nB = self.n_batch
        nx = self.prob.n_state
        nu = self.prob.n_ctrl
        assert Q_.shape == torch.Size([nB, nx, nx])
        assert q_.shape == torch.Size([nB, nx])
        assert R_.shape == torch.Size([nB, nu, nu])
        assert r_.shape == torch.Size([nB, nu])
        assert K_.shape == torch.Size([nB, nu, nx])
        assert k_.shape == torch.Size([nB, nu])
        assert V_.shape == torch.Size([nB, nx, nx])
        assert v_.shape == torch.Size([nB, nx])

        return V_, v_, K_, k_, Q_, q_, R_, r_

    def riccati_forward_(self, Dx0, K, k, A, B, b):
        Du = mv(K, Dx0) + k
        Dx = mv(A, Dx0) + mv(B, Du) + b

        return Dx, Du
