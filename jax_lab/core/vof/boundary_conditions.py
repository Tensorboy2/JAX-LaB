"""
Boundary conditions specific to the volume-of-fluid (VoF) lattice Boltzmann model.
"""

from ..boundary_conditions import NonEquilibriumExtrapolation


class VoFOpenBoundary(NonEquilibriumExtrapolation):
    """
    Open (constant-pressure reservoir) boundary for the gas lattice of the VoF model.

    The density is fixed at the prescribed reservoir value while the velocity is taken from the adjacent interior node (zero-gradient),
    using non-equilibrium extrapolation for the missing populations, so gas produced at the interface is advected out of the domain
    without a pressure drift.

    It is also the switch for evaporation: a nonzero evaporation_flux enables the interfacial mass-transfer source of the VoF solver
    (paper Eqs. 16-22). Without a VoFOpenBoundary carrying a nonzero flux, the solver compiles no evaporation term at all.

    Parameters
    ----------
    indices (tuple): Boundary node indices.

    grid_info (dict): Grid information of the simulation.

    precision_policy (PrecisionPolicy): Precision policy of the simulation.

    prescribed (float, optional): Reservoir density in gas lattice units. Defaults to 1.0.

    evaporation_flux (float, optional): Evaporation mass flux per unit interfacial area, positive for liquid loss, in liquid lattice units
        (J dt / (rho_liquid dx)). Defaults to 0.0, which disables evaporation.

    evaporation_start (int, optional): Timestep at which evaporation is switched on. Defaults to 0.

    References
    ----------
    1. Mohammadrezaei, S., Farajzadeh, R. & Niasar, V. Modeling the drying process of CO2-water in porous media with a novel
    volume-of-fluid lattice Boltzmann model. Water Resources Research 62, e2025WR040829 (2026).

    2. Zhao-Li, G., Chu-Guang, Z. & Bao-Chang, S. Non-equilibrium extrapolation method for velocity and pressure boundary conditions in the lattice
    Boltzmann method. Chinese Phys. 11, 366-374 (2002).
    """

    def __init__(self, indices, grid_info, precision_policy, prescribed=1.0, evaporation_flux=0.0, evaporation_start=0):
        super().__init__(indices, grid_info, precision_policy, prescribed)
        self.name = "VoFOpenBoundary"
        if evaporation_flux < 0.0:
            raise ValueError("evaporation_flux must be non-negative (positive values remove liquid).")
        self.evaporation_flux = float(evaporation_flux)
        self.evaporation_start = int(evaporation_start)
