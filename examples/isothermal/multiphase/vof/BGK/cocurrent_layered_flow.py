"""
Co-current layered two-phase flow in a channel (Mohammadrezaei et al. 2026, Section 3.1.1): density ratio 1000:1, kinematic viscosity
ratio 1:10, both layers driven by the same pressure gradient. The steady velocity profile is compared with the analytical two-layer
Poiseuille solution.
"""

import os

import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.utils import save_fields_hdf5_xdmf
from jax_lab.core.vof import VoFBGK

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def analytical_velocity(y, W, G, mu1, mu2):
    """
    Two-layer Poiseuille flow with -dp/dx = G, walls at y = 0 and y = W and fluid 1 below y = W / 2 (velocity and shear stress continuous):
    u_k = G W^2 / (8 mu_k) [2 mu_k / (mu1 + mu2) + (mu1 - mu2) / (mu1 + mu2) s - s^2], s = 2 y / W - 1.
    Eq. 27 of the paper has the same structure but, as printed, does not vanish at the walls.
    """
    s = 2.0 * y / W - 1.0
    mu = np.where(y <= W / 2, mu1, mu2)
    return G * W**2 / (8.0 * mu) * (2.0 * mu / (mu1 + mu2) + (mu1 - mu2) / (mu1 + mu2) * s - s**2)


class CocurrentLayeredFlow(VoFBGK):
    def initialize_volume_fraction(self):
        # Walls (full-way bounce-back nodes y = 0 and ny - 1) sit at y = 0.5 and ny - 1.5, so cell j spans [j - 1, j] in channel coordinates
        y = np.arange(self.ny) - 0.5
        eps = np.clip(W / 2 - (y - 0.5), 0.0, 1.0)
        return np.broadcast_to(eps, (self.nx, self.ny))

    def set_boundary_conditions(self):
        walls = np.concatenate((self.bounding_box_indices["bottom"], self.bounding_box_indices["top"]))
        for BCs in self.BCs:
            BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy))

    def output_data(self, **kwargs):
        u = kwargs["u"][0, 0, 1:-1, 0]
        y = np.arange(1, self.ny - 1) - 0.5
        u_exact = analytical_velocity(y, W, G, mu1, mu2)
        error = np.sqrt(np.mean((u - u_exact) ** 2)) / np.max(np.abs(u_exact))
        logger.info(f"Timestep {kwargs['timestep']}: relative RMS error = {error:.4e}, max u = {u.max():.4e} (exact {u_exact.max():.4e})")
        np.savetxt(
            os.path.join(output_dir, f"profile_{kwargs['timestep']:07d}.csv"), np.column_stack((y, u, u_exact)), delimiter=",", header="y,u,u_exact"
        )
        plot_profile(y, u, kwargs["timestep"], error)
        fields = {
            "volume_fraction": kwargs["volume_fraction"][0, ..., 0],
            "flags": kwargs["flags"][0, ..., 0],
            "p": kwargs["p"][0, ..., 0],
            "ux": kwargs["u"][0, ..., 0],
            "uy": kwargs["u"][0, ..., 1],
        }
        save_fields_hdf5_xdmf(kwargs["timestep"], fields, output_dir, prefix="fields")


def plot_profile(y, u, timestep, error):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_fine = np.linspace(0.0, W, 400)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.axhspan(0.0, 0.5, color="tab:blue", alpha=0.12, label="liquid")
    ax.axhspan(0.5, 1.0, color="tab:orange", alpha=0.12, label="gas")
    ax.plot(analytical_velocity(y_fine, W, G, mu1, mu2) / analytical_velocity(W / 2, W, G, mu1, mu2), y_fine / W, "k-", label="analytical")
    ax.plot(u / analytical_velocity(W / 2, W, G, mu1, mu2), y / W, "o", ms=3.5, mfc="none", color="tab:red", label="VoF-LBM")
    ax.set_xlabel(r"$u / u_{interface}$")
    ax.set_ylabel(r"$y / W$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Co-current layered flow, step {timestep}, relative RMS error {100 * error:.2f}%", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "profile.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    precision = "f32/f32"
    W = 129
    # The flow is uniform along x; the channel is W long only to give the XDMF output a visible extent
    nx, ny = W, W + 2

    density_ratio = 1.0e-3
    nu_l, nu_g = 0.05, 0.5
    mu1, mu2 = nu_l, density_ratio * nu_g  # dynamic viscosities in liquid units
    # -dp/dx, the same force per unit volume in both layers, scaled with 1 / W^2 to keep the peak velocity (Mach number) fixed
    G = 1.0e-7 * (65 / W) ** 2
    # Diffusive time to steady state grows as W^2
    t_max = int(200000 * (W / 65) ** 2)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_cocurrent_layered_flow")
    os.makedirs(output_dir, exist_ok=True)

    kwargs = {
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "precision": precision,
        "omega": [1.0 / (3.0 * nu_l + 0.5), 1.0 / (3.0 * nu_g + 0.5)],
        "density_ratio": density_ratio,
        # Accelerations differ by 1 / density_ratio so that both layers feel the same pressure gradient
        "body_force": [[G, 0.0], [G / density_ratio, 0.0]],
        "io_rate": t_max // 10,
        "print_info_rate": t_max // 10,
    }
    sim = CocurrentLayeredFlow(**kwargs)
    sim.run(t_max)
