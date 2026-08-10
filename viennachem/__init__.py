# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lado Filipovic, Institute for Microelectronics, TU Wien
"""ViennaChem - surface-kinetics mechanisms derived from reaction equations.

A reaction file lists species, their phases, the surface reactions and their
rate constants. Everything else is derived: the free sites (from site
conservation), the stoichiometry, the rate laws and their exponents, the
coverage balance and its Jacobian, and the surface velocity.

    import viennachem as vc

    data = vc.from_file("silane.yaml")        # mechanism data, as a dict
    vc.write("silane.mechanism.json", data)   # what a solver consumes
    theta, v = vc.evaluate(data)              # the reference answer

The library knows nothing about any particular simulator. ViennaPS reads the
mechanism data in C++ and runs it as `ps.SurfaceChemistry`; anything else with a
surface-kinetics solver could read the same file.
"""

from .emit import (
    SCHEMA_VERSION,
    build_data,
    from_file,
    write,
)
from .evaluate import (
    constant_for,
    coverage_sites,
    default_fluxes,
    drate_dtheta,
    evaluate,
    free_fractions,
    growth_rate,
    ion_channel_flux,
    mechanism_fluxes,
    rate,
    rate_constants,
    solid_density,
    solve_coverages,
    sticking,
)
from .mechanism import (
    ENERGY_UNITS,
    EV_PER_KCAL_PER_MOL,
    KB_EV,
    Mechanism,
    REACTIONS_DIR,
    atoms_of,
    molar_mass,
    resolve,
    speed_coefficient,
)

__all__ = [
    # the compiler
    "SCHEMA_VERSION",
    "Mechanism",
    "REACTIONS_DIR",
    "ENERGY_UNITS",
    "EV_PER_KCAL_PER_MOL",
    "KB_EV",
    "atoms_of",
    "build_data",
    "from_file",
    "molar_mass",
    "speed_coefficient",
    "resolve",
    "write",
    # the reference evaluator: everything a solver has to reproduce
    "constant_for",
    "coverage_sites",
    "default_fluxes",
    "drate_dtheta",
    "evaluate",
    "free_fractions",
    "growth_rate",
    "ion_channel_flux",
    "mechanism_fluxes",
    "rate",
    "rate_constants",
    "solid_density",
    "solve_coverages",
    "sticking",
]
