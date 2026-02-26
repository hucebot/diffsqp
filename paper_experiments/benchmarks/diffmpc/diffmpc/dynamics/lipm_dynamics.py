"""LIPM dynamics class."""

from typing import Any, Dict

import jax.numpy as jnp

from diffmpc.dynamics.base_dynamics import Dynamics

lipm_parameters: Dict[str, Any] = {
    "num_states": 4,
    "num_controls": 2,
    "names_states": ["r_x", "r_y", "rdot_x", "rdot_y"],
    "names_controls": ["z_x", "z_y"],
}
lipm_state_dot_parameters = None


class LipmDynamics(Dynamics):
    """LIPM dynamics class."""

    def __init__(self, parameters: Dict[str, Any] = lipm_parameters):
        """
        Initializes the class.
        Args:
            parameters:  parameters of the class.
                (str, Any) dictionary
        """
        super().__init__(parameters)

    def state_dot(
        self,
        state: jnp.array,
        control: jnp.array,
        state_dot_params: Dict[str, Any] = lipm_state_dot_parameters,
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
        w2 = state_dot_params["omega2"]
        dt = state_dot_params["discretization_resolution"]
        # Extract states (no checks are done here)
        r = jnp.array(state[0:2])
        rdot = jnp.array(state[2:4])
        z = control

        rddot = w2 * (r - z)

        state_dot = jnp.concatenate((rdot, rddot))
        return state_dot
