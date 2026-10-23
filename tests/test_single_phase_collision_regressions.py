"""Regression test for the single-phase cascaded collision."""

import jax.numpy as jnp
import numpy as np

from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.models import CLBMSim


PRECISION = "f32/f32"


def _base_parameters():
    return {
        "lattice": LatticeD2Q9(PRECISION),
        "omega": 1.0,
        "nx": 4,
        "ny": 4,
        "nz": 0,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
    }


def test_cascaded_collision_uses_relaxed_moments():
    matrix = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 1, 1, 1, 1, 2, 2, 2, 2],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 1, 1, -1, -1],
        [0, 0, 0, 0, 0, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])
    sim = CLBMSim(
        **_base_parameters(),
        M=matrix,
        s_0=1.0,
        s_1=1.0,
        s_b=1.0,
        s_2=1.0,
        s_3=1.0,
        s_4=1.0,
    )
    fin = jnp.broadcast_to(sim.w, (4, 4, sim.q)).at[..., 1].add(0.01)

    fout = sim.collision(fin)

    assert np.max(np.abs(np.asarray(fout - fin))) > 1e-4
    np.testing.assert_allclose(np.asarray(jnp.sum(fout, axis=-1)), np.asarray(jnp.sum(fin, axis=-1)), atol=1e-5)
