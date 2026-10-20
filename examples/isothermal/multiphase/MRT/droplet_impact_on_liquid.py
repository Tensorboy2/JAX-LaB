"""
Droplet impact on a thin liquid film (Yang & Qin 2024, Sec. 4.4), using the fourth-order EFM pseudopotential force with the
mesh/momentum-space scale factor lambda.

Paper setup: 1000 x 1000 x 500 lu, droplet radius R = 60, film thickness h = 25, impact velocity U = 0.11 along -z, periodic in x
and y, no-slip in z, Peng-Robinson EOS at Tr = 0.5 (density ratio 11074), lambda = 0.1, tau_1^-1 = 0.5, Re = 17000 and
1 - kappa = 0.65 (We = 379). The paper's grid does not fit on two 24 GB GPUs, so the droplet is scaled by R / 60 while U, lambda and
Tr are kept (We ~ 200 for R = 32). The film is 35 lu, thicker relative to the droplet than in the paper (h / D ~ 0.55 vs 0.21).

Deviations needed for stability: s_e = 0.1 instead of 0.5 (bulk viscosity damps the start-up
compression wave that otherwise drives the liquid into the EOS pole at rho = 1/b), s_eps = s_q = 0.9 instead of 1.1, and
tau_v = 0.52 (Re ~ 1060) instead of the paper's 0.5023. With s_eps = s_q = 1.0, tau_v = 0.52 diverges when the vapor behind the crown
is rarefied to zero density (the mesh-space vapor sound speed is only ~ lambda * sqrt(RT) ~ 0.02); with 0.9, tau_v <= 0.515 still
diverges. We * sqrt(Re) ~ 6500 is above the splashing threshold of 3000.

Walls are three solid layers thick so the two-cell EFM gradient stencil never reaches across the domain.

References:
1. Yang, Zhichao, and Zhangrong Qin. "A three-dimensional pseudo-potential multiphase model with super-large density ratio and
adjustable surface tension." Computers and Fluids 278, 106318 (2024). https://doi.org/10.1016/j.compfluid.2024.106318.
2. d'Humieres, D. et al. Multiple-relaxation-time lattice Boltzmann models in three dimensions. Philosophical Transactions of the
Royal Society of London. Series A 360, 437-451 (2002).
"""

import logging
import os
import subprocess

import jax.numpy as jnp
import numpy as np

from jax_lab.core.boundary_conditions import BounceBackHalfway
from jax_lab.core.eos import PengRobinson
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseMRT
from jax_lab.core.utils import save_fields_hdf5_xdmf

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

WALL_LAYERS = 3


class DropletOnLiquid3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x, y, z = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij")
        film_top = WALL_LAYERS - 0.5 + h
        film = 0.5 * (1.0 - np.tanh(2.0 * (z - film_top) / width))
        # Droplet sits one interface width above the film so the two interfaces start separated
        z_center = film_top + r + width
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - z_center) ** 2)
        drop = 0.5 * (1.0 - np.tanh(2.0 * (dist - r) / width))

        rho = rho_g + (rho_l - rho_g) * np.clip(film + drop, 0.0, 1.0)
        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u[..., 2] = -U * drop

        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho[..., None])
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
        return [self.precision_policy.cast_to_output(rho)], [self.precision_policy.cast_to_output(u)]

    def set_boundary_conditions(self):
        layers = list(range(WALL_LAYERS)) + list(range(self.nz - WALL_LAYERS, self.nz))
        x, y, z = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.array(layers), indexing="ij")
        walls = (x.ravel(), y.ravel(), z.ravel())
        self.BCs[0].append(BounceBackHalfway(walls, self.grid_info, self.precision_policy))

    def apply_contact_angle(self, rho_tree):
        """Neutral (90 degree) virtual wall density: copy the nearest fluid layer into every solid layer, so the two-cell EFM
        stencil sees a zero normal gradient at the walls instead of the periodic z seam between the liquid film and the vapor."""
        n = WALL_LAYERS
        return [rho.at[:, :, :n].set(rho[:, :, n : n + 1]).at[:, :, -n:].set(rho[:, :, -n - 1 : -n]) for rho in rho_tree]

    def compute_fluid_fluid_force(self, psi_tree):
        """Interaction force on fluid nodes only; solid nodes carry no force."""
        fluid = np.ones((1, 1, self.nz, 1), dtype=np.float32)
        fluid[:, :, :WALL_LAYERS] = 0.0
        fluid[:, :, -WALL_LAYERS:] = 0.0
        return [force * jnp.asarray(fluid, dtype=force.dtype) for force in super().compute_fluid_fluid_force(psi_tree)]

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ..., 0])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        logger.info(f"t* = {U * timestep / (2 * r):.2f}  rho: [{rho.min():.3e}, {rho.max():.4f}]  |u|max = {np.sqrt((u**2).sum(-1)).max():.3e}")
        fields = {"rho": rho, "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}
        save_fields_hdf5_xdmf(timestep, fields, output_dir, "data")


if __name__ == "__main__":
    e = LatticeD3Q19().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((19, 19))
    M[0, :] = en**0
    M[1, :] = 19 * en**2 - 30
    M[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    M[3, :] = e[:, 0]
    M[4, :] = (5 * en**2 - 9) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (5 * en**2 - 9) * e[:, 1]
    M[7, :] = e[:, 2]
    M[8, :] = (5 * en**2 - 9) * e[:, 2]
    M[9, :] = 3 * e[:, 0] ** 2 - en**2
    M[10, :] = (3 * en**2 - 5) * (3 * e[:, 0] ** 2 - en**2)
    M[11, :] = e[:, 1] ** 2 - e[:, 2] ** 2
    M[12, :] = (3 * en**2 - 5) * (e[:, 1] ** 2 - e[:, 2] ** 2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1] ** 2 - e[:, 2] ** 2) * e[:, 0]
    M[17, :] = (e[:, 2] ** 2 - e[:, 0] ** 2) * e[:, 1]
    M[18, :] = (e[:, 0] ** 2 - e[:, 1] ** 2) * e[:, 2]

    # Droplet radius scaled from the paper's R = 60
    r = 32
    h = 35  # thicker than the paper's 25 lu (not scaled with the droplet)
    nx, ny, nz = 500, 500, 120
    width = 5
    U = 0.11

    # Peng-Robinson EOS at Tr = 0.5; coexistence densities from Maxwell's equal-area construction
    a = 2 / 49
    b = 2 / 21
    R = 1.0
    pr_omega = 0.344
    Tc = 0.07292041977847488
    T = 0.5 * Tc
    rho_l = 9.21933196568944
    rho_g = 0.0008326124686554373
    eos = PengRobinson(a=a, b=b, R=R, pr_omega=pr_omega, T=T)

    # Paper: tau_v = 0.5 + 3 * U * 120 / 17000 (Re = 17000 at R = 60); 0.52 is the lowest value found here that survives the splash
    tau_v = 0.52
    logger.info(f"Re = {U * 2 * r / ((tau_v - 0.5) / 3):.0f}, film h = {h}, steps per unit t* = {2 * r / U:.0f}")

    output_dir = os.path.abspath("./droplet_impact")
    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "lambda_": [0.1],
        "s_rho": [0.0],
        "s_e": [0.1],
        "s_eta": [0.9],
        "s_j": [0.0],
        "s_q": [0.9],
        "s_pi": [1.0],
        "s_m": [1.0],
        "s_v": [1.0 / tau_v],
        "M": [M],
        "kappa": [0.35],  # 1 - kappa = 0.65
        "precision": precision,
        "io_rate": 100,
        "compute_MLUPS": False,  # True disables all output
        "print_info_rate": 100,
        "checkpoint_rate": 2000,  # only the latest checkpoint is kept; restart with "restore_checkpoint": True
        "checkpoint_dir": os.path.abspath("./droplet_impact_checkpoints"),
        "restore_checkpoint": False,
    }

    subprocess.run(["rm", "-rf", output_dir], check=True)
    sim = DropletOnLiquid3D(**kwargs)
    # 10000 steps ~ t* = 17.2 (t* = U t / D); output every 100 steps ~ 0.17 t*
    sim.run(10000)
