"""
Water droplet resting on a curved (convex, circular) solid surface in a CO2 atmosphere, without gravity. The droplet relaxes from a
semicircle to the circular arc set by the prescribed contact angle; the steady contact angle is measured from a circle fitted to the
liquid-gas interface. For a droplet arc of radius R_d and a solid circle of radius R_s whose centers are a distance d apart, the contact
angle through the liquid is cos(theta) = (R_s^2 + R_d^2 - d^2) / (2 R_s R_d).

Fluid properties follow the water-scCO2 pair of Mohammadrezaei et al. (2026, Table 2): density ratio 1 : 2.56, kinematic viscosity
ratio 7.56 : 1.
"""

import os

import contourpy
import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.vof import VoFMRT

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def fit_circle(points):
    # Algebraic least-squares circle: x^2 + y^2 + a x + b y + c = 0
    x, y = points.T
    a, b, c = np.linalg.lstsq(np.column_stack((x, y, np.ones_like(x))), -(x**2 + y**2), rcond=None)[0]
    center = np.array([-a / 2, -b / 2])
    return center, np.sqrt(center @ center - c)


def measure_contact_angle(eps):
    lines = contourpy.contour_generator(z=eps.T).lines(0.5)
    points = np.concatenate(lines)
    # Keep the free interface only: drop the contour along the solid and the contact-line region
    points = points[np.linalg.norm(points - cylinder_center, axis=1) > R_surface + exclusion]
    center, radius = fit_circle(points)
    d = np.linalg.norm(center - cylinder_center)
    cos_theta = (R_surface**2 + radius**2 - d**2) / (2.0 * R_surface * radius)
    return np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0))), center, radius


class DropletOnCylinder(VoFMRT):
    def initialize_volume_fraction(self):
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        eps = np.clip(R0 + 0.5 - np.hypot(x - droplet_center[0], y - droplet_center[1]), 0.0, 1.0)
        return np.where(solid, 0.0, eps)

    def set_boundary_conditions(self):
        walls = np.argwhere(solid)
        for BCs in self.BCs:
            BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy, theta=np.deg2rad(contact_angle)))

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        eps = kwargs["volume_fraction"][0, ..., 0]
        theta, center, radius = measure_contact_angle(eps)
        u_max = np.max(np.linalg.norm(kwargs["u"][0], axis=-1))
        history.append((timestep, theta, radius, u_max))
        logger.info(
            f"Timestep {timestep}: contact angle = {theta:.2f} deg (prescribed {contact_angle:.0f}), R_d = {radius:.2f}, max |u| = {u_max:.2e}"
        )
        shapes[contact_angle] = (eps, center, radius)


def plot_results(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 4))
    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    for ax, (angle, theta, _) in zip(axes, results, strict=False):
        eps, center, radius = shapes[angle]
        ax.imshow(np.where(solid, np.nan, eps).T, origin="lower", cmap="Blues", vmin=0.0, vmax=1.5)
        ax.add_patch(plt.Circle(cylinder_center, R_surface, color="0.55"))
        ax.plot(center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi), "r--", lw=1, label="fitted arc")
        ax.set_xlim(0, nx - 1)
        ax.set_ylim(cylinder_center[1] - 10, ny - 1)
        ax.set_aspect("equal")
        ax.set_title(f"prescribed {angle:.0f} deg, measured {theta:.1f} deg", fontsize=9)
        ax.legend(loc="lower left", fontsize=7)
    prescribed, measured = np.array([(a, t) for a, t, _ in results]).T
    axes[-1].plot([40, 140], [40, 140], "k-", lw=1, label="measured = prescribed")
    axes[-1].plot(prescribed, measured, "o", color="tab:red", label="VoF-LBM")
    axes[-1].set_xlabel("prescribed contact angle (deg)")
    axes[-1].set_ylabel("measured contact angle (deg)")
    axes[-1].set_aspect("equal")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "contact_angles.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    precision = "f32/f32"
    nx, ny = 200, 200

    R_cylinder = 45
    cylinder_center = np.array([nx / 2, 60.0])
    # Full-way bounce-back: the effective surface lies halfway between the solid and the fluid nodes
    R_surface = R_cylinder + 0.5
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    solid = np.hypot(x - cylinder_center[0], y - cylinder_center[1]) <= R_cylinder

    R0 = 30
    droplet_center = cylinder_center + np.array([0.0, R_surface])
    exclusion = 4.0

    density_ratio = 1.0 / 2.56
    nu_l = 0.2
    nu_g = nu_l / 7.56
    sigma = 0.01
    t_max = 60000

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

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_droplet_on_cylinder_2d")
    os.makedirs(output_dir, exist_ok=True)
    shapes = {}
    results = []

    for contact_angle in (60.0, 90.0, 120.0):
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
            "io_rate": 5000,
            "print_info_rate": 20000,
        }
        sim = DropletOnCylinder(**kwargs)
        sim.run(t_max)
        np.savetxt(
            os.path.join(output_dir, f"history_{contact_angle:.0f}.csv"),
            np.array(history),
            delimiter=",",
            header="timestep,contact_angle,droplet_radius,max_u",
        )
        results.append((contact_angle, history[-1][1], history[-1][2]))

    for angle, theta, radius in results:
        logger.info(f"Prescribed {angle:.0f} deg: steady contact angle {theta:.2f} deg (error {theta - angle:+.2f} deg), R_d = {radius:.2f}")
    np.savetxt(os.path.join(output_dir, "contact_angles.csv"), np.array(results), delimiter=",", header="prescribed,measured,droplet_radius")
    plot_results(results)
