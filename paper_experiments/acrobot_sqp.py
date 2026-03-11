import json
import argparse
import torch

from diffsqp.problems import Problem
from diffsqp.costs import LqrCost, TerminalCost
from diffsqp.dynamics import AcrobotInverseDynamics
from diffsqp.solvers import Lqr
from diffsqp.solvers import Ssqp


def main(args):
    # Set default device (cpu or gpu)
    torch.set_default_device(args.dev)

    if args.model == "forward":
        # dyn = CartPoleDynamics(mc=1.0, mp=0.1, lp=0.5, grav=9.81)
        dyn = None
    elif args.model == "inverse":
        dyn = AcrobotInverseDynamics(
            m1=1.0,
            m2=1.0,
            l1=0.5,
            l2=0.5,
            lc1=0.5,
            lc2=0.5,
            grav=9.81,
            I2=1 / (3.0 * 1.0 * 0.5**2),
            I1=1 / (3.0 * 1.0 * 0.5**2),
        )
    else:
        print("ERROR: Unsupported dynamics model. Supported are forward, inverse")
        exit()

    dt = 0.01
    tf = 1.0
    horizon = int(tf / dt)
    n_batch = args.nb
    n_state = dyn.nx
    n_ctrl = dyn.nu

    x_des = torch.tensor([torch.pi, 0.0, 0.0, 0.0]).repeat(n_batch, 1)

    x_init = torch.tensor([torch.pi, 0.0, 0.0, 0.0]).repeat(n_batch, 1)
    x_init[:, 0:2] += args.std * torch.randn((n_batch, 2))

    prob = Problem(horizon, dt, n_state, n_ctrl)

    # Set stage cost and constraints
    q_w = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6])
    r_w = torch.tensor([1e-1])
    qf_w = torch.tensor([4e8, 4e8, 1e5, 1e5])
    Q = q_w * torch.eye(n_state).repeat(n_batch, 1, 1)
    R = r_w * torch.eye(n_ctrl).repeat(n_batch, 1, 1)
    Qf = qf_w * torch.eye(n_state).repeat(n_batch, 1, 1)

    for i in range(horizon - 1):
        if i == 0:
            prob.states.append(x_init.clone())
        else:
            prob.states.append(x_des.clone())
        prob.controls.append(torch.zeros((n_batch, n_ctrl)))
        prob.costs.append(LqrCost(Q, R, x_des.clone()))
        prob.stage_dynamics.append(dyn)
    # Set terminal cost
    prob.states.append(x_des.clone())
    prob.costs.append(TerminalCost(Qf, x_des.clone()))

    # Select internal QP solver
    qp_solver = Lqr(prob)
    # Create solver object
    solver = Ssqp(prob, qp_solver)
    info = solver.solve()

    info["err_final_state"] = torch.abs(torch.square(prob.states[-1] - x_des)).tolist()
    print("Final state error: ", info["err_final_state"])
    # Save solver logs to json
    with open("log.json", "w") as f:
        json.dump(info, f, indent=4)

    # Save solution (states and controls)
    if args.save:
        torch.save(prob.states, "states.pt")
        torch.save(prob.controls, "controls.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", type=int, help="Batch size")
    parser.add_argument("-model", type=str, help="Dynamics model: forward, inverse")
    parser.add_argument("-dev", type=str, help="Device to solve: cpu, cuda")
    parser.add_argument(
        "-std", type=float, help="Initial state noise standard deviation"
    )
    parser.add_argument("-save", action="store_true", help="Save solution for viz")
    main(parser.parse_args())
