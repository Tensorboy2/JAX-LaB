"""
Shan-Chen (pseudopotential) multiphase coupling of the hybrid thermal LBM solver, with phase change driven by the
temperature-dependent EOS of the fluid solver. See jax_lab.core.thermal for the shared finite difference solver and references.
"""

import operator
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.tree import map as tree_map
from jax.tree import reduce

from ..thermal import Thermal


class MultiphaseThermal(Thermal):
    """
    Multiphase implementation of the hybrid thermal LBM solver with phase
    change support. The fluid is advanced by the wrapped Multiphase solver
    (pytrees of distribution functions, one per component) while a single
    shared temperature field is advanced by the finite difference solver.

    The temperature equation solved is (reference 1 in the module docstring):
    \\frac{\\partial T}{\\partial t} = -\\mathbf{u} \\cdot \\nabla T + \\frac{1}{\\rho c_v}(\\nabla \\cdot (K \\nabla T)
    - T \\frac{\\partial p_{EOS}}{\\partial T} \\nabla \\cdot \\mathbf{u} + S_T)

    where the T (dp_EOS/dT) div(u) term accounts for the latent heat released or absorbed during phase change, rho and u are the mixture density
    and velocity, and S_T is a user defined source term (see source()).

    The temperature field is passed into the fluid solver every step, so the EOS of the fluid solver must be constructed with
    temperature_field_type="thermal"; the pressure (and hence the pseudopotential) then follows the local temperature, which is what drives
    evaporation and condensation.

    Assumes all components share a single temperature field (thermal equilibrium between components at every node).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.fluid_solver.eos.temperature_field_type != "thermal":
            raise ValueError('The EOS of the fluid solver must be constructed with temperature_field_type="thermal" for MultiphaseThermal.')
        if self.apply_buoyancy:
            raise ValueError("apply_buoyancy is not supported for MultiphaseThermal; use the body_force parameter of the fluid solver instead.")

    @partial(jit, static_argnums=(0,), inline=True)
    def divergence_x(self, u):
        """
        Compute the divergence of a vector field using the isotropic lattice
        difference stencil, by summing the diagonal entries of the gradient of
        each velocity component.

        Parameters
        ----------
        u (jax.numpy.ndarray): Vector field of shape (nx, ny, 2) in 2D or (nx, ny, nz, 3) in 3D.

        Returns
        -------
        jax.numpy.ndarray: Divergence of the field, of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.
        """
        div_u = jnp.zeros_like(u[..., :1])
        for alpha in range(self.dim):
            div_u = div_u + self.grad_x(u[..., alpha : alpha + 1])[..., alpha : alpha + 1]
        return div_u

    @partial(jit, static_argnums=(0,), inline=True)
    def RHS(self, T, rho_tree, u_tree):
        """
        Right hand side of the thermal equation for multiphase flow:
        -\\mathbf{u} \\cdot \\nabla T + \\frac{1}{\\rho c_v}(K \\nabla^2 T + \\nabla K \\cdot \\nabla T
        - T \\frac{\\partial p_{EOS}}{\\partial T} \\nabla \\cdot \\mathbf{u} + S_T)

        The mixture (total) density and velocity are used; the phase change
        term uses the sum of dp_EOS/dT over all components.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.

        rho_tree (pytree of jax.numpy.ndarray): Density field of all components.

        u_tree (pytree of jax.numpy.ndarray): Velocity field of all components.

        Returns
        -------
        jax.numpy.ndarray: RHS of the thermal equation, of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.
        """
        grad_T = self.grad_x(T)
        lap_T = self.laplacian_x(T)
        # Mixture (total) density and velocity
        rho = self.fluid_solver.compute_total_density(rho_tree)
        u = self.fluid_solver.compute_total_velocity(rho_tree, u_tree)

        advection = jnp.sum(u * grad_T, axis=-1, keepdims=True)
        diffusion = self.K * lap_T + jnp.sum(self.grad_K * grad_T, axis=-1, keepdims=True)

        div_u = self.divergence_x(u)
        dp_eos_dT_tree = self.fluid_solver.eos.dp_eos_dT(rho_tree, T)
        phase_change = T * reduce(operator.add, dp_eos_dT_tree) * div_u

        return -advection + (diffusion - phase_change + self.source(T)) / (rho * self.c_v)

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1, 2))
    def step(self, T_prev, f_poststreaming_tree, timestep, return_fpost=False):
        """
        This function performs a single step of the multiphase hybrid thermal LBM simulation.

        It evaluates the component fields and temperature right hand side at
        the current time level, then advances the fluid and temperature fields
        together by one time step. The current temperature is passed to the
        thermal EOS used by the fluid collision.

        Parameters
        ----------
        T_prev (jax.numpy.ndarray): The temperature field from the previous timestep.

        f_poststreaming_tree (pytree of jax.numpy.ndarray): The post-streaming distribution functions.

        timestep (int): The current timestep of the simulation.

        return_fpost (bool, optional): If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        T (jax.numpy.ndarray): The temperature field after the simulation step.

        f_poststreaming_tree (pytree of jax.numpy.ndarray): The post-streaming distribution functions after the simulation step.

        f_postcollision_tree (pytree of jax.numpy.ndarray or None): The post-collision distribution functions after the
        simulation step, or None if return_fpost is False.
        """
        T = self.precision_policy.cast_to_compute(T_prev)
        T = self.apply_bc(T, timestep)
        f_compute_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), f_poststreaming_tree)
        rho_tree, _ = self.fluid_solver.update_macroscopic(f_compute_tree)
        u_tree = self.fluid_solver.macroscopic_velocity(f_compute_tree, rho_tree, T=T)

        f_poststreaming_tree, f_postcollision_tree = self.fluid_solver.step(f_poststreaming_tree, timestep, return_fpost, T)
        T = self._advance_temperature(T, timestep, rho_tree, u_tree)

        T = self.apply_bc(T, timestep)

        return self.precision_policy.cast_to_output(T), f_poststreaming_tree, f_postcollision_tree

    def update_macroscopic_output(self, f_tree, T):
        """
        Compute the component densities and force-corrected velocities for I/O.

        Parameters
        ----------
        f_tree (pytree of jax.numpy.ndarray): Post-streaming distribution functions.

        T (jax.numpy.ndarray): Temperature field.

        Returns
        -------
        rho_tree (pytree of jax.numpy.ndarray): Density field of all components.

        u_tree (pytree of jax.numpy.ndarray): Velocity field of all components.
        """
        f_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), f_tree)
        rho_tree, _ = self.fluid_solver.update_macroscopic(f_tree)
        u_tree = self.fluid_solver.macroscopic_velocity(f_tree, rho_tree, T=self.precision_policy.cast_to_compute(T))
        return rho_tree, u_tree
