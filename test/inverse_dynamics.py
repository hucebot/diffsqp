import torch
from diffsqp.dynamics import (
    Dynamics,
    CartPoleInverseDynamics,
    CartPoleInverseDynamicsConstrained,
    AcrobotInverseDynamics,
)


def test_dynamics_derivatives(
    dyn: Dynamics, x: torch.Tensor, u: torch.Tensor, dt: float
):
    n_batch = x.shape[0]

    # 1. Get analytical Jacobian
    fx_analytic = dyn.fx(x, u, dt)
    fu_analytic = dyn.fu(x, u, dt)
    gx_analytic = dyn.gx(x, u)
    gu_analytic = dyn.gu(x, u)

    # 2. Get numerical/autograd Jacobian
    # We define a wrapper to handle the batch dimension for autograd
    def fx_wrapper(x_in):
        return dyn.f(x_in, u, dt)

    def fu_wrapper(u_in):
        return dyn.f(x, u_in, dt)

    def gx_wrapper(x_in):
        return dyn.g(x_in, u)

    def gu_wrapper(u_in):
        return dyn.g(x, u_in)

    # Compute Jacobian for each element in the batch
    fx_numeric = []
    fu_numeric = []
    gx_numeric = []
    gu_numeric = []
    for i in range(n_batch):
        # jacobian returns (output_dim, input_dim)
        jx = torch.autograd.functional.jacobian(fx_wrapper, x)[i, :, i, :]
        ju = torch.autograd.functional.jacobian(fu_wrapper, u)[i, :, i, :]
        gx = torch.autograd.functional.jacobian(gx_wrapper, x)[i, :, i, :]
        gu = torch.autograd.functional.jacobian(gu_wrapper, u)[i, :, i, :]
        fx_numeric.append(jx)
        fu_numeric.append(ju)
        gx_numeric.append(gx)
        gu_numeric.append(gu)
    fx_numeric = torch.stack(fx_numeric)
    fu_numeric = torch.stack(fu_numeric)
    gx_numeric = torch.stack(gx_numeric)
    gu_numeric = torch.stack(gu_numeric)

    # 3. Assert dimensions
    assert dyn.f(x, u, dt).shape == (n_batch, dyn.nx)
    assert fx_analytic.shape == (n_batch, dyn.nx, dyn.nx)
    assert fu_analytic.shape == (n_batch, dyn.nx, dyn.nu)
    assert gx_analytic.shape == (n_batch, dyn.ng, dyn.nx)
    assert gu_analytic.shape == (n_batch, dyn.ng, dyn.nu)

    # 4. Compare derivatives
    torch.set_printoptions(2)
    assert torch.allclose(fx_analytic, fx_numeric, atol=1e-6)
    assert torch.allclose(fu_analytic, fu_numeric, atol=1e-6)
    assert torch.allclose(gx_analytic, gx_numeric, atol=1e-6)
    assert torch.allclose(gu_analytic, gu_numeric, atol=1e-6)

    print("Dynamics tests passed!")


if __name__ == "__main__":
    n_batch = 3
    dt = 0.01
    # dyn = CartPoleInverseDynamics(mc=1.0, mp=1.0, lp=1.0, grav=9.81)
    # dyn = AcrobotInverseDynamics(m1=0.1, m2=0.1, l1=0.3, l2=0.3)
    dyn = CartPoleInverseDynamicsConstrained(
        mc=1.0, mp=1.0, lp=1.0, constr_u=True, grav=9.81
    )

    x = torch.randn(n_batch, dyn.nx, requires_grad=True)
    u = torch.randn(n_batch, dyn.nu, requires_grad=True)

    test_dynamics_derivatives(dyn, x, u, dt)
