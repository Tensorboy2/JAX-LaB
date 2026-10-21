"""
Droplet impact on a liquid film (Yang & Qin 2024, Sec. 4.4), using the fourth-order EFM pseudopotential force with the
mesh/momentum-space scale factor lambda. Periodic in x and y, no-slip walls in z, Peng-Robinson EOS at Tr = 0.6.

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
        # float32 with broadcast 1D axes (no full 3D coordinate grids); the droplet is evaluated only in its bounding box
        f32 = np.float32
        x = np.arange(self.nx, dtype=f32)[:, None, None]
        y = np.arange(self.ny, dtype=f32)[None, :, None]
        z = np.arange(self.nz, dtype=f32)[None, None, :]
        film_top = WALL_LAYERS - 0.5 + h
        phase = np.empty((self.nx, self.ny, self.nz), dtype=f32)  # liquid fraction, 0 (vapor) to 1 (liquid)
        phase[...] = 0.5 * (1.0 - np.tanh(2.0 * (z - film_top) / width))

        # The droplet's lower edge starts a distance gap above the film surface
        center = (self.nx / 2, self.ny / 2, film_top + gap + r)
        reach = r + 4.0 * width  # the tanh profile is below 1e-6 beyond this distance
        box = tuple(slice(max(int(c - reach), 0), min(int(c + reach) + 2, n)) for c, n in zip(center, phase.shape))
        dist = np.sqrt((x[box[0]] - center[0]) ** 2 + (y[:, box[1]] - center[1]) ** 2 + (z[:, :, box[2]] - center[2]) ** 2)
        drop = 0.5 * (1.0 - np.tanh(2.0 * (dist - r) / width))
        phase[box] = np.clip(phase[box] + drop, 0.0, 1.0)

        u = np.zeros((self.nx, self.ny, self.nz, 3), dtype=f32)
        u[box + (2,)] = -U * drop
        rho = phase
        rho *= rho_l - rho_g
        rho += rho_g

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

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ..., 0])
        # u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        # logger.info(f"t* = {U * timestep / (2 * r):.2f}  rho: [{rho.min():.3e}, {rho.max():.4f}]  |u|max = {np.sqrt((u**2).sum(-1)).max():.3e}")
        logger.info(f"t* = {U * timestep / (2 * r):.2f}  rho: [{rho.min():.3e}, {rho.max():.4f}]")
        # fields = {"rho": rho, "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}
        fields = {"rho": rho}
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

    r = 40
    h = 17.0  # film thickness
    gap = 10  # release height of the droplet's lower edge above the film
    nx, ny, nz = 640, 640, 150
    width = 8
    U = 0.11

    # Peng-Robinson EOS at Tr = 0.6 (density ratio 853, close to water-air); coexistence densities from Maxwell's equal-area construction
    a = 2 / 49
    b = 2 / 21
    R = 1.0
    pr_omega = 0.344
    Tc = 0.07292041977847488
    T = 0.6 * Tc
    rho_l = 8.724919319823417
    rho_g = 0.010227924611067207
    eos = PengRobinson(a=a, b=b, R=R, pr_omega=pr_omega, T=T)

    tau_v = 0.52
    logger.info(
        f"Re = {U * 2 * r / ((tau_v - 0.5) / 3):.0f}, film h = {h}, gap = {gap}, impact at step ~ {gap / U:.0f}, steps per unit t* = {2 * r / U:.0f}"
    )

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
        "lambda_": [0.2],
        "s_rho": [0.0],
        "s_e": [0.1],
        "s_eta": [0.9],
        "s_j": [0.0],
        "s_q": [0.9],
        "s_pi": [1.0],
        "s_m": [1.0],
        "s_v": [1.0 / tau_v],
        "M": [M],
        "kappa": [0.8],
        "precision": precision,
        "io_rate": 200,
        "compute_MLUPS": False,  # True disables all output
        "print_info_rate": 100,
        "checkpoint_rate": 1000,  # only the latest checkpoint is kept; restart with "restore_checkpoint": True
        "checkpoint_dir": os.path.abspath("./droplet_impact_checkpoints"),
        "restore_checkpoint": True,
    }

    if not kwargs["restore_checkpoint"]:  # keep the earlier frames when continuing from a checkpoint
        subprocess.run(["rm", "-rf", output_dir], check=True)
    sim = DropletOnLiquid3D(**kwargs)
    # 7000 steps ~ t* = 9.6 (t* = U t / D); output every 200 steps ~ 0.28 t*
    sim.run(7000)
