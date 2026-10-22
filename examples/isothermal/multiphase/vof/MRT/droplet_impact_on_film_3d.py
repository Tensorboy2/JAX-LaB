"""
Impact of a water droplet on a thin water film in air, under gravity. The droplet of diameter D falls at speed U onto a film of thickness
h = 0.2 D resting on a no-slip wall and forms a crown whose rim develops a Rayleigh-Taylor (fingering) instability. The dimensionless groups
are

    Re = U D / nu_l = 10000, We = rho_l U^2 D / sigma = 1000, Fr = U / sqrt(g D) = 36.44, Ca = mu_l U / sigma = We / Re = 0.1,
    Bo = rho_l g D^2 / sigma = We / Fr^2 = 0.753,

so that the splashing parameter We^0.8 Re^0.4 (about 10000) is well above the crown-splash threshold of about 2100. The air-water density
ratio is 1 : 1000 and the kinematic viscosity ratio 15 : 1. The droplet starts just above the film.
"""

import os

import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.utils import save_fields_hdf5_xdmf
from jax_lab.core.vof import VoFMRT

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


class DropletImpact(VoFMRT):
    def initialize_volume_fraction(self):
        x, y, z = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij")
        droplet = np.clip(D / 2 + 0.5 - np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2 + (z - z_c) ** 2), 0.0, 1.0)
        film = np.clip(film_top - z + 0.5, 0.0, 1.0)
        eps = np.maximum(droplet, film)
        eps[:, :, 0] = eps[:, :, -1] = 0.0
        return eps

    def initialize_macroscopic_fields(self):
        x, y, z = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij")
        inside = (x - x_c) ** 2 + (y - y_c) ** 2 + (z - z_c) ** 2 <= (D / 2 + 1.0) ** 2
        rho = np.ones((self.nx, self.ny, self.nz, 1))
        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u[inside, 2] = -U
        return [rho, rho], [u, np.zeros_like(u)]

    def set_boundary_conditions(self):
        walls = np.concatenate((self.bounding_box_indices["bottom"], self.bounding_box_indices["top"]))
        for BCs in self.BCs:
            BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy))

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        eps = kwargs["volume_fraction"][0, ..., 0]
        u = kwargs["u"][0]
        logger.info(
            f"Timestep {timestep} (t U / D = {timestep * U / D:.2f}): liquid volume = {eps.sum():.0f}, max |u| / U = {np.abs(u).max() / U:.2f}"
        )
        fields = {"volume_fraction": eps, "p": kwargs["p"][0, ..., 0], "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}
        save_fields_hdf5_xdmf(timestep, fields, output_dir, prefix="impact")


if __name__ == "__main__":
    precision = "f32/f32"

    Re, We, Fr = 10000.0, 1000.0, 36.44
    D = 56
    U = 0.1
    nu_l = U * D / Re
    nu_g = 15.0 * nu_l
    density_ratio = 1.0e-3
    sigma = U**2 * D / We
    g = U**2 / (Fr**2 * D)

    nx = ny = 6 * D
    nz = 9 * D // 2
    film_top = 0.5 + 0.2 * D
    x_c, y_c = nx / 2, ny / 2
    z_c = film_top + D / 2 + 2.0

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
    s_v = [1.0 / (3.0 * nu_l + 0.5), 1.0 / (3.0 * nu_g + 0.5)]

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_droplet_impact_on_film_3d")
    os.makedirs(output_dir, exist_ok=True)

    kwargs = {
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": precision,
        "M": [M, M],
        "s_rho": [0.0, 0.0],
        "s_e": [1.0, 1.0],
        "s_eta": [1.0, 1.0],
        "s_j": [0.0, 0.0],
        "s_q": [1.0, 1.0],
        "s_v": s_v,
        "s_pi": [1.0, 1.0],
        "s_m": [1.0, 1.0],
        "density_ratio": density_ratio,
        "surface_tension": sigma,
        "body_force": [0.0, 0.0, -g],
        "io_rate": 200,
        "print_info_rate": 200,
    }
    sim = DropletImpact(**kwargs)
    sim.run(3500)
