"""Nonlinear Acrobot dynamics (continuous time)."""

from typing import Any, Dict

import jax.numpy as jnp

from diffmpc.dynamics.base_dynamics import Dynamics

default_parameters: Dict[str, Any] = {
    "num_states": 4,  # [th1, th2, thd1, thd2]
    "num_controls": 1,  # [torque]
    "names_states": ["th1", "th2", "thd1", "thd2"],
    "names_controls": ["u"],
}


class AcrobotDynamics(Dynamics):
    """
    Continuous-time cart-pole dynamics using the standard formulation.

    State:  [th1, thd1, th2, thd2]
    Control: [u]
    """

    def __init__(self, parameters: Dict[str, Any] = default_parameters):
        super().__init__(parameters)

    def state_dot(
        self,
        state: jnp.array,
        control: jnp.array,
        params: Dict[str, Any],
    ) -> jnp.array:
        """
        Computes the time derivative of the state of the system.

        Returns x_dot = f(x, u) where f describes the dynamics of the system.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            params: parameters of the state_dot function of the dynamics.
                (str, Any) dictionary

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        th1, th2, thd1, thd2 = state
        tau = control[0]

        m1 = params["m1"]
        m2 = params["m2"]
        l1 = params["l1"]
        l2 = params["l2"]
        lc1 = params["lc1"]
        lc2 = params["lc2"]
        I1 = params["I1"]
        I2 = params["I2"]
        grav = params["grav"]

        s1 = jnp.sin(th1)
        s2 = jnp.sin(th2)
        c2 = jnp.cos(th2)
        s12 = jnp.sin(th1 + th2)

        denom = I1 * I2 + I2 * l1**2 * m2 - l1**2 * lc2**2 * m2**2 * c2**2
        nom1 = -I2 * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * thd1 + thd2) * s2 * thd2
        )
        nom2 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * thd1**2 - tau
        )
        thdd1 = (nom1 + nom2) / denom

        # denom = I1 * I2 + I2 * l1**2 * m2 - l1**2 * lc2**2 * m2**2 * c2**2
        nom1 = (I2 + l1 * lc2 * m2 * c2) * (
            grav * l1 * m2 * s1
            + grav * lc1 * m1 * s1
            + grav * lc2 * m2 * s12
            - l1 * lc2 * m2 * (2.0 * thd1 + thd2) * s2 * thd2
        )
        nom2 = -(grav * lc2 * m2 * s12 + l1 * lc2 * m2 * s2 * thd2 * thd2 - tau) * (
            I1 + I2 + l1 * l1 * m2 + 2.0 * l1 * lc2 * m2 * c2
        )
        thdd2 = (nom1 + nom2) / denom

        return jnp.array([thd1, thd2, thdd1, thdd2])
