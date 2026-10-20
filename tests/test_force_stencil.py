"""Verify the EFM multiphase force stencil against periodic NumPy references."""

import numpy as np

from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseBGK

DOMAIN_3D = (12, 12, 12)
DOMAIN_2D = (16, 16, 0)
PRECISION = "f32/f32"
SEED = 0


def _reference_efm_gradient(field, c, w, dim):
    """Fourth-order isotropic EFM gradient: sum_i 3 w_i c_i (4/3 f(x + c_i) - 1/6 f(x + 2 c_i)), periodic rolls."""
    total = np.zeros((*field.shape[:-1], dim))
    for direction, weight in zip(c.T, w, strict=True):
        if not np.any(direction):
            continue
        near = np.roll(field[..., 0], tuple(-direction), axis=tuple(range(dim)))
        far = np.roll(field[..., 0], tuple(-2 * direction), axis=tuple(range(dim)))
        total += 3.0 * weight * ((4.0 / 3.0) * near - (1.0 / 6.0) * far)[..., None] * direction
    return total


def _reference_compute_fluid_fluid_force(psi_arrays, g_kkprime, c, w, dim):
    gradients = [_reference_efm_gradient(psi, c, w, dim) for psi in psi_arrays]
    n = len(psi_arrays)
    return [-psi_arrays[k] * sum(g_kkprime[k, j] * gradients[j] for j in range(n)) for k in range(n)]


def _check(domain, lattice_class):
    dim = 2 if domain[2] == 0 else 3
    nx, ny, nz = domain
    lattice = lattice_class(PRECISION)

    g_kkprime = np.array([[-1.0, -0.5], [-0.5, -1.0]])

    kwargs = {
        "lattice": lattice,
        "omega": [1.0, 1.0],
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
        "n_components": 2,
        "g_kkprime": g_kkprime,
        "lambda_": [1.0, 1.0],
        "EOS": VanderWaals(a=[9.0 / 49.0, 9.0 / 49.0], b=[2.0 / 21.0, 2.0 / 21.0], R=[1.0, 1.0], T=0.8 * 0.5714285714),
    }
    sim = MultiphaseBGK(**kwargs)

    spatial_shape = domain[:2] if dim == 2 else domain
    rng = np.random.default_rng(SEED + 1)
    psi_host = [rng.uniform(0.1, 1.0, size=(*spatial_shape, 1)).astype(np.float32) for _ in range(2)]
    psi_tree = [sim.distributed_array_init(p.shape, sim.precision_policy.compute_dtype, init_val=p) for p in psi_host]

    actual = [np.asarray(f) for f in sim.compute_fluid_fluid_force(psi_tree)]

    c = np.asarray(lattice.c)
    w = np.asarray(lattice.w, dtype=np.float64)
    expected = _reference_compute_fluid_fluid_force(psi_host, g_kkprime, c, w, dim)

    for a, e in zip(actual, expected, strict=True):
        assert np.max(np.abs(a - e)) < 1e-5


def test_compute_fluid_fluid_force_matches_reference_3d():
    _check(DOMAIN_3D, LatticeD3Q19)


def test_compute_fluid_fluid_force_matches_reference_2d():
    _check(DOMAIN_2D, LatticeD2Q9)
