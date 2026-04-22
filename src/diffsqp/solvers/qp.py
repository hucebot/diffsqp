import torch

from diffsqp.problems import Problem


class QP:
    def __init__(self, prob: Problem) -> None:
        self.prob = prob
        self.horizon = self.prob.horizon
        self.n_B = self.prob.states[0].shape[0]
        self.n_x = self.prob.nx
        self.n_u = self.prob.nu
        self.nvars = (self.horizon - 1) * (self.n_x + self.n_u) + self.n_x

        self.delta_x = [None] * self.horizon
        self.delta_u = [None] * (self.horizon - 1)
        # Lagrange multipliers of the actuation part
        self.mu = [None] * (self.horizon)
        # Lagrange multipliers of the underactuation part
        self.nu = [None] * (self.horizon - 1)

    def solve(self):
        Q_qp, p_qp, G_qp, h_qp, A_qp, b_qp = self.generate_problem_()

        sol, lam = self.solve_kkt_(Q_qp, p_qp, A_qp, b_qp)

        for i in range(self.horizon - 1):
            start = i * (self.n_x + self.n_u)
            self.delta_x[i] = sol[:, start : start + self.n_x]
            self.delta_u[i] = sol[:, start + self.n_x : start + self.n_x + self.n_u]
            self.mu[i + 1] = lam[:, i * self.n_x : i * self.n_x + self.n_x]
        self.delta_x[-1] = sol[:, -self.n_x :]

        # Return corrections
        return self.delta_x, self.delta_u, self.mu, self.nu

    def generate_problem_(self):
        n_B = self.n_B
        nhor = self.horizon
        n_x = self.n_x
        n_u = self.n_u
        nvars = self.nvars
        I = torch.eye(n_x).repeat(n_B, 1, 1)
        Q_qp = torch.zeros((n_B, nvars, nvars))
        p_qp = torch.zeros((n_B, nvars))
        A_qp1 = torch.zeros((n_B, (nhor - 1) * n_x, nvars))
        b_qp1 = torch.zeros((n_B, (nhor - 1) * n_x))

        A_qp2 = None
        b_qp2 = None
        if self.prob.constraints[0]:
            ng = self.prob.constraints[0].ng
            A_qp2 = torch.zeros((n_B, (nhor - 1) * ng, nvars))
            b_qp2 = torch.zeros((n_B, (nhor - 1) * ng))
        for i in range(self.horizon - 1):
            x_lin = self.prob.states[i]
            u_lin = self.prob.controls[i]
            x_next = self.prob.states[i + 1]

            # Create cost matrices
            Q, R, S, q, r = self.calc_linearized_cost_terms_(i, x_lin, u_lin)
            ST = torch.transpose(S, 1, 2)

            d_idx = i * (n_x + n_u)
            Q_qp[:, d_idx : d_idx + n_x, d_idx : d_idx + n_x] = Q
            Q_qp[:, d_idx : d_idx + n_x, d_idx + n_x : d_idx + n_x + n_u] = ST
            Q_qp[:, d_idx + n_x : d_idx + n_x + n_u, d_idx : d_idx + n_x] = S
            Q_qp[
                :, d_idx + n_x : d_idx + n_x + n_u, d_idx + n_x : d_idx + n_x + n_u
            ] = R

            p_qp[:, d_idx : d_idx + n_x] = q
            p_qp[:, d_idx + n_x : d_idx + n_x + n_u] = r

            # Create constraint matrices
            A, B, b, C, D, e = self.calc_linearized_dynamic_terms_(
                x_lin, u_lin, x_next, self.prob.dynamics[i]
            )
            c_i = i * (n_x + n_u)
            r_i = i * n_x
            A_qp1[:, r_i : r_i + n_x, c_i : c_i + n_x] = A
            A_qp1[:, r_i : r_i + n_x, c_i + n_x : c_i + n_x + n_u] = B
            A_qp1[:, r_i : r_i + n_x, c_i + n_x + n_u : c_i + n_x + n_u + n_x] = -I

            b_qp1[:, r_i : r_i + n_x] = -b

            if self.prob.constraints[0]:
                ng = self.prob.dynamics[0].ng
                c_i = i * (n_x + n_u)
                r_i = i * ng
                A_qp2[:, r_i : r_i + ng, c_i : c_i + n_x] = C
                A_qp2[:, r_i : r_i + ng, c_i + n_x : c_i + n_x + n_u] = D
                b_qp2[:, r_i : r_i + ng] = -e

        x_F = self.prob.states[-1]
        Q_F = self.prob.lxx(-1, x_F)
        q_F = self.prob.lx(-1, x_F)
        Q_qp[:, -n_x:, -n_x:] = Q_F
        p_qp[:, -n_x:] = q_F

        A_qp1[:, -n_x:, -n_x:] = -I

        # Initial state constraint
        Is = torch.zeros((n_B, n_x, nvars))
        Is[:, 0:n_x, 0:n_x] = torch.eye(n_x)
        A_qp1 = torch.cat([Is, A_qp1], dim=1)
        b_qp1 = torch.cat([torch.zeros((n_B, n_x)), b_qp1], dim=1)

        A_qp = None
        b_qp = None
        if self.prob.constraints[0]:
            A_qp = torch.cat([A_qp1, A_qp2], dim=1)
            b_qp = torch.cat([b_qp1, b_qp2], dim=1)
        else:
            A_qp = A_qp1
            b_qp = b_qp1

        # torch.set_printoptions(precision=2, linewidth=1000)
        # print(A_qp2[0])
        # print(b_qp2[0])
        # exit()

        G_qp = torch.zeros((n_B, 1, nvars))
        G_qp[:, 0, 0] = 1.0
        h_qp = 1e6 * torch.ones((n_B, 1))

        # Q_qp += 10 * torch.eye(nvars).repeat(n_B, 1, 1)
        # Q_qp = self.ensure_psd(Q_qp)
        return Q_qp, p_qp, G_qp, h_qp, A_qp, b_qp

    # def ensure_psd(self, Q, eps=1e-6, verbose=True):
    #     # 1. Symmetrize to fix any numerical noise (QP matrices must be symmetric)
    #     Q_sym = (Q + Q.transpose(-1, -2)) / 2
    #
    #     # 2. Compute eigenvalues and eigenvectors
    #     # torch.linalg.eigh is optimized for Hermitian/symmetric matrices
    #     L, V = torch.linalg.eigh(Q_sym)
    #
    #     # 3. Check for negative eigenvalues
    #     min_eig = L.min(dim=-1)[0]  # Min eigenvalue per batch
    #
    #     if (min_eig < -eps).any():
    #         if verbose:
    #             print(
    #                 f"Warning: Matrix not PSD. Min eigenvalue: {min_eig.min().item():.2e}. Fixing..."
    #             )
    #
    #         # 4. Clamp eigenvalues to be at least eps
    #         # This projects the matrix onto the PSD cone (closest PSD matrix in Frobenius norm)
    #         L_new = torch.clamp(L, min=eps)
    #
    #         # 5. Reconstruct the matrix
    #         Q_psd = V @ torch.diag_embed(L_new) @ V.transpose(-1, -2)
    #         return Q_psd
    #
    #     return Q_sym

    def calc_linearized_cost_terms_(self, stage_idx, x_lin, u_lin):
        Q = self.prob.lxx(stage_idx, x_lin, u_lin)
        q = self.prob.lx(stage_idx, x_lin, u_lin)
        R = self.prob.luu(stage_idx, x_lin, u_lin)
        r = self.prob.lu(stage_idx, x_lin, u_lin)
        S = self.prob.lux(stage_idx, x_lin, u_lin)
        return Q, R, S, q, r

    def calc_linearized_dynamic_terms_(self, x_lin, u_lin, x_next, dynamics):
        x_pred = dynamics.f(x_lin, u_lin, self.prob.dt)
        b = x_pred - x_next
        A = dynamics.fx(x_lin, u_lin, self.prob.dt)
        B = dynamics.fu(x_lin, u_lin, self.prob.dt)
        C = None
        D = None
        e = None
        if self.prob.constraints[0]:
            C = dynamics.gx(x_lin, u_lin)
            D = dynamics.gu(x_lin, u_lin)
            e = dynamics.g(x_lin, u_lin)
        return A, B, b, C, D, e

    def solve_kkt_(self, Q, q, A, b):
        n_B, n, _ = Q.shape
        m = A.shape[1]

        zeros = torch.zeros((n_B, m, m), device=Q.device, dtype=Q.dtype)
        top_row = torch.cat([Q, A.transpose(1, 2)], dim=2)  # Result: (B, n, n+m)
        bottom_row = torch.cat([A, zeros], dim=2)  # Result: (B, m, n+m)
        KKT_matrix = torch.cat([top_row, bottom_row], dim=1)  # Result: (B, n+m, n+m)

        rhs = torch.cat([-q, b], dim=1)  # Result: (B, n+m)

        solution = torch.linalg.solve(KKT_matrix, rhs)

        x = solution[:, :n]
        lam = solution[:, n:]

        return x, lam
