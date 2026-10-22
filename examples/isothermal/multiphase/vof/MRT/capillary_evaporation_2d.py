"""
Evaporation from two connected capillaries (Mohammadrezaei et al. 2026, Section 3.1.2), in 2D with two parallel-plate slits of half-widths
b1 < b2 joined at the bottom. Both start at the same liquid level; capillarity lifts the liquid in the narrow slit until the level difference
reaches Jurin's law for slits, dH = sigma cos(theta) / (drho g) (1 / b1 - 1 / b2). Evaporation is then switched on at both menisci through
the open top boundary, and dH must stay at its equilibrium value while both levels drop.

Fluid properties follow the paper's water-scCO2 pair (Table 2): density ratio 1 : 2.56, kinematic viscosity ratio 7.56 : 1.
"""

import os

import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.vof import VoFMRT, VoFOpenBoundary

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


class CapillaryEvaporation2D(VoFMRT):
    def initialize_volume_fraction(self):
        eps = np.zeros((self.nx, self.ny))
        eps[:, :h0] = 1.0
        return eps

    def set_boundary_conditions(self):
        theta = np.deg2rad(contact_angle)
        walls = np.argwhere(solid)
        top = np.array([[x, self.ny - 1] for x in range(self.nx) if not solid[x, self.ny - 1]])
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
        # Meniscus height on the slit centerlines, the liquid column being the integral of the volume fraction
        h1 = eps[x1_center, 1:].sum() + 0.5
        h2 = eps[x2_center, 1:].sum() + 0.5
        history.append((timestep, h1, h2, h1 - h2))
        error = (h1 - h2 - dH_jurin) / dH_jurin
        logger.info(f"Timestep {timestep}: h1 = {h1:.2f}, h2 = {h2:.2f}, dH = {h1 - h2:.2f}, Jurin dH = {dH_jurin:.2f}, error = {100 * error:.2f}%")
        np.savetxt(os.path.join(output_dir, "heights.csv"), np.array(history), delimiter=",", header="timestep,h1,h2,dH")
        plot_heights(np.array(history))
        if timestep % (10 * self.io_rate) == 0:
            fields = {"volume_fraction": eps, "p": kwargs["p"][0, ..., 0], "ux": kwargs["u"][0, ..., 0], "uy": kwargs["u"][0, ..., 1]}
            save_fields_vtk(timestep, fields, output_dir, "data")


def plot_heights(history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, h1, h2, dH = history.T
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax1.plot(t, h1, color="tab:blue", label=f"narrow slit (b = {b1})")
    ax1.plot(t, h2, color="tab:orange", label=f"wide slit (b = {b2})")
    ax1.set_ylabel("meniscus height")
    ax1.legend(fontsize=8)
    ax2.plot(t, dH, color="tab:red", label="VoF-LBM")
    ax2.axhline(dH_jurin, color="k", ls="--", label="Jurin")
    ax2.set_ylabel(r"$\Delta H = h_1 - h_2$")
    ax2.set_xlabel("timestep")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.axvline(evaporation_start, color="0.6", lw=0.8, ls=":")
    ax1.set_title(f"Capillary rise, then evaporation from step {evaporation_start}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heights.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    precision = "f32/f32"

    b1, b2 = 6, 24  # slit half-widths
    separator = 8
    nx = 1 + 2 * b1 + separator + 2 * b2 + 1
    ny = 260
    connector = 24  # height of the bottom channel joining both slits
    h0 = 120

    solid = np.zeros((nx, ny), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = True
    solid[1 + 2 * b1 : 1 + 2 * b1 + separator, connector:] = True
    x1_center = 1 + b1
    x2_center = 1 + 2 * b1 + separator + b2

    density_ratio = 1.0 / 2.56
    nu_l = 0.2
    nu_g = nu_l / 7.56
    sigma = 0.01
    contact_angle = 40.0
    g = 5.0e-5
    dH_jurin = sigma * np.cos(np.deg2rad(contact_angle)) / ((1.0 - density_ratio) * g) * (1.0 / b1 - 1.0 / b2)

    evaporation_start = 250000
    evaporation_flux = 1.0e-4
    t_max = 450000

    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)
    M = np.zeros((9, 9))
    M[0, :] = en**0
    M[1, :] = -4 * en**0 + 3 * en**2
    M[2, :] = 4 * en**0 - (21 / 2) * en**2 + (9 / 2) * en**4
    M[3, :] = e[:, 0]
    M[4, :] = (-5 * en**0 + 3 * en**2) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (-5 * en**0 + 3 * en**2) * e[:, 1]
    M[7, :] = e[:, 0] ** 2 - e[:, 1] ** 2
    M[8, :] = e[:, 0] * e[:, 1]
    s_v = [1.0 / (3.0 * nu_l + 0.5), 1.0 / (3.0 * nu_g + 0.5)]

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_capillary_evaporation_2d")
    os.makedirs(output_dir, exist_ok=True)
    history = []

    kwargs = {
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "precision": precision,
        "M": [M, M],
        "s_rho": [0.0, 0.0],
        "s_e": [1.0, 1.0],
        "s_eta": [1.0, 1.0],
        "s_j": [0.0, 0.0],
        "s_q": [1.0, 1.0],
        "s_v": s_v,
        "density_ratio": density_ratio,
        "surface_tension": sigma,
        "body_force": [0.0, -g],
        "io_rate": 5000,
        "print_info_rate": 50000,
    }
    sim = CapillaryEvaporation2D(**kwargs)
    sim.run(t_max)
