import torch
from diffsqp.costs import TerminalCost


def test_terminal_cost_derivatives(cost: TerminalCost):
    # Helper to compute Jacobian of a gradient w.r.t a variable
    def get_jacobian(grad, var):
        # Flattens the gradient and variable to compute a standard Jacobian matrix
        nB = grad.shape[0]
        grad_flat = grad.view(nB, -1)
        var_flat = var.view(nB, -1)

        jac = []
        for i in range(grad_flat.shape[1]):
            grad_out = torch.zeros_like(grad_flat)
            grad_out[:, i] = 1
            # retain_graph=True allows us to call backward multiple times
            j_col = torch.autograd.grad(
                grad_flat,
                var,
                grad_outputs=grad_out,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if j_col is None:
                j_col = torch.zeros((nB, 1, var.shape[1]))
            jac.append(j_col.view(nB, 1, -1))
        return torch.cat(jac, dim=1)

    # Create random inputs with gradients enabled
    x = torch.randn(nB, nx, requires_grad=True)

    # Compute analytical values from your class
    lx_analytic = cost.lx(x)
    lxx_analytic = cost.lxx(x)

    # Compute numerical gradients using autograd
    # First Gradients
    l_sum = cost.l(x).sum()
    l_sum.backward(create_graph=True)

    lx_numeric = x.grad
    lxx_numeric = get_jacobian(x.grad, x)

    # Assert shapes
    assert cost.l(x).shape == (nB,)
    assert lx_analytic.shape == (nB, nx)
    assert lxx_analytic.shape == (nB, nx, nx)

    # Final values
    assert torch.allclose(lx_analytic, x.grad)
    assert torch.allclose(lxx_analytic, lxx_numeric, atol=1e-6)

    print("Terminal cost tests passed!")


if __name__ == "__main__":
    nB = 2
    nx = 3
    Q = torch.rand(nB, nx, 1) * torch.eye(nx).repeat(nB, 1, 1)
    x_des = torch.ones((nB, nx))

    cost = TerminalCost(Q, x_des)

    test_terminal_cost_derivatives(cost)
