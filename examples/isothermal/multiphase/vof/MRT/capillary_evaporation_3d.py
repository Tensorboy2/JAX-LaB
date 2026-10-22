"""
Evaporation from two connected capillary tubes (Mohammadrezaei et al. 2026, Section 3.1.2 and Appendix B): two vertical tubes of radii
R1 < R2 joined at the bottom by a reservoir. Both start at the same liquid level; capillarity lifts the liquid in the narrow tube until the
level difference reaches Jurin's law (Eq. 31), dH = 2 sigma cos(theta) / (drho g) (1 / R1 - 1 / R2). Evaporation is then switched on at both
menisci through the open top boundary, and dH must stay at its equilibrium value while both levels drop.

Fluid properties follow the paper's water-scCO2 pair (Table 2): density ratio 1 : 2.56, kinematic viscosity ratio 7.56 : 1.
"""

import os

import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.vof import VoFMRT, VoFOpenBoundary

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


class CapillaryEvaporation3D(VoFMRT):
    def initialize_volume_fraction(self):
        eps = np.zeros((self.nx, self.ny, self.nz))
        eps[:, :, :h0] = 1.0
        return eps

    def set_boundary_conditions(self):
        theta = np.deg2rad(contact_angle)
        walls = np.argwhere(solid)
        top = np.argwhere(~solid[:, :, -1])
        top = np.column_stack((top, np.full(len(top), self.nz - 1)))
        for BCs in self.BCs:
            BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy, theta=theta))
        self.BCs[1].append(
            VoFOpenBoundary(
                tuple(top.T),
                self.grid_info,
                self.precision_policy,
                prescribed=1.0,
                evaporation_flux=evaporation_flux,
                evaporation_start=evaporation_start,
            )
        )

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        eps = kwargs["volume_fraction"][0, ..., 0]
        # Meniscus height on the tube axes, the liquid column being the integral of the volume fraction
        h1 = eps[x1_center, y_center, 1:].sum() + 0.5
        h2 = eps[x2_center, y_center, 1:].sum() + 0.5
        history.append((timestep, h1, h2, h1 - h2))
        error = (h1 - h2 - dH_jurin) / dH_jurin
        logger.info(f"Timestep {timestep}: h1 = {h1:.2f}, h2 = {h2:.2f}, dH = {h1 - h2:.2f}, Jurin dH = {dH_jurin:.2f}, error = {100 * error:.2f}%")
        np.savetxt(os.path.join(output_dir, "heights.csv"), np.array(history), delimiter=",", header="timestep,h1,h2,dH")
        if timestep % (10 * self.io_rate) == 0:
            fields = {"volume_fraction": eps, "p": kwargs["p"][0, ..., 0], "uz": kwargs["u"][0, ..., 2]}
            save_fields_vtk(timestep, fields, output_dir, "data")


if __name__ == "__main__":
    precision = "f32/f32"

    R1, R2 = 6, 18  # tube radii
    gap = 6
    nx = 2 + 2 * R1 + gap + 2 * R2
    ny = 2 + 2 * R2
    nz = 180
    connector = 16  # height of the bottom reservoir joining both tubes
    h0 = 80

    x1_center, x2_center, y_center = 1 + R1, 1 + 2 * R1 + gap + R2, ny // 2
    x, y = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="ij")
    tubes = ((x - x1_center - 0.5) ** 2 + (y - y_center - 0.5) ** 2 < R1**2) | ((x - x2_center - 0.5) ** 2 + (y - y_center - 0.5) ** 2 < R2**2)
    solid = np.ones((nx, ny, nz), dtype=bool)
    solid[1:-1, 1:-1, 1:connector] = False
    solid[:, :, connector:] = ~tubes[..., None]

    density_ratio = 1.0 / 2.56
    nu_l = 0.2
    nu_g = nu_l / 7.56
    sigma = 0.01
    contact_angle = 40.0
    g = 1.0e-4
    dH_jurin = 2.0 * sigma * np.cos(np.deg2rad(contact_angle)) / ((1.0 - density_ratio) * g) * (1.0 / R1 - 1.0 / R2)

    evaporation_start = 250000
    evaporation_flux = 1.0e-4
    t_max = 450000

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

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_capillary_evaporation_3d")
    os.makedirs(output_dir, exist_ok=True)
    history = []

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
        "io_rate": 5000,
        "print_info_rate": 50000,
    }
    sim = CapillaryEvaporation3D(**kwargs)
    sim.run(t_max)
