"""Manipulator dynamics class."""

from typing import Any, Dict

import jax.numpy as jnp

from diffmpc.dynamics.base_dynamics import Dynamics

import jax
import mujoco
from mujoco import mjx

manipulator_parameters: Dict[str, Any] = {}
manipulator_state_dot_parameters = None


class ManipulatorDynamics(Dynamics):
    """Manipulator dynamics class."""

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initializes the class.
        Args:
            parameters:  parameters of the class.
                (str, Any) dictionary
        """

        self.forward_fn = jax.jit(mjx.forward)

        mj_model = mujoco.MjModel.from_xml_path(parameters["xml_path"])
        mj_data = mujoco.MjData(mj_model)
        self.mjx_model = mjx.put_model(mj_model)
        self.mjx_data = mjx.put_data(mj_model, mj_data)
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        parameters["num_states"] = self.nq + self.nv
        parameters["num_controls"] = self.nu
        parameters["names_states"] = [f"q_{i}" for i in range(self.nq)]
        parameters["names_states"] += [f"dq_{i}" for i in range(self.nv)]
        parameters["names_controls"] = [f"u_{i}" for i in range(self.nu)]
        super().__init__(parameters)

    def state_dot(
        self,
        state: jnp.array,
        control: jnp.array,
        state_dot_params: Dict[str, Any] = manipulator_state_dot_parameters,
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
        qpos = jnp.array(state[0 : self.nq])
        qvel = jnp.array(state[self.nq : self.nq + self.nv])

        data = self.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=control)

        data = mjx.forward(self.mjx_model, data)

        state_dot = jnp.concatenate((data.qvel, data.qacc))
        return state_dot
