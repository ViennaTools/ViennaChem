#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lado Filipovic, Institute for Microelectronics, TU Wien
"""Compile a reaction file into mechanism data.

The mechanism data is the published contract between this library and whatever
solves the chemistry. It carries exponents and indices, never expressions, so a
consumer with no symbolic algebra can rebuild both the residual and its Jacobian
from it. ViennaPS reads it in C++ (`psChemicalMechanismIO.hpp`).

`schemaVersion` is the compatibility handle: a reader that does not know a
version must refuse the file rather than guess at it.

Schema
------
{
  "schemaVersion": int,                          # the format's compatibility handle
  "name": str,
  "siteTypes": [ {"name": str, "density": float|null} ],
  "coverages": [ {"name": str, "site": int} ],
  "gas":       [ {"name": str, "traced": bool, "label": str, "flux": float|null,
                  "sticking": {...}|null } ],
  "solids":    [ {"name": str, "rho": float,     # each solid phase and its density
                  "rhoMaterials": {mat: float}} ], # densities that differ per material
  "solid":     solids[0],                        # for readers expecting one solid
  "reactions": [ {"id": str, "eq": str,
                  "constant":    {"kind": "sticking"|"arrhenius", "k0"|"s0", "Ea",
                                  "beta"},   # k = pre * T^beta * exp(-Ea/kT)
                  "gasFactors":  [ {"idx": int, "exp": int} ],
                  "covFactors":  [ {"idx": int, "exp": int} ],
                  "freeSiteExp": [int],          # one entry per site type
                  "nu":          [float],        # one entry per coverage
                  "solidAtoms":  [float] } ],    # atoms of each solid added per event
  "constants": {"temperature": float},
  "conventions": {"stickingAppliedIn": "rateLawAndReflection", "energyUnit": "eV"}
}

Sticking convention (design risk 1)
-----------------------------------
An adsorption reaction's constant *is* the sticking coefficient. It is applied
once in the rate law (analytically, in the coverage balance) and once in the
particle's re-emission during transport. The particle records the *raw* incident
flux. Applying it anywhere else counts it twice.

Site density (design risk 2)
----------------------------
`density` belongs to a site type, because a multi-site mechanism has one density
per type and their ratio matters. It is optional, and the steady-state solve does
not use it: with theta a fraction and r an areal rate, the balance is

    S0 * dtheta_i/dt = sum_j nu_ij r_j

so S0 cancels when the left side is zero. It also does not enter the velocity,
which uses the solid density `solid.rho`. A density is required only for

  - transient (non-steady-state) coverage integration,
  - converting a per-site attempt frequency nu [1/s] into an areal k0 = S0 * nu,
  - multiple site types, where the ratio of densities enters.

Supply it in the reaction file as a top-level `siteTypes:` block, or as `S0` on a
surface species when there is a single type.

Usage:  python3 -m viennachem silane.yaml [out.mechanism.json]
"""

import json
import sys

from . import mechanism

KB_EV = mechanism.KB_EV

# Bumped whenever the meaning or the shape of the emitted data changes.
SCHEMA_VERSION = 1


def _density_of(mech, tname):
    """Density of site type `tname`, or None when it was not supplied.

    Looked for by name in the top-level `siteTypes:` block, then (for a single
    type) as `S0` on a surface species. Absence is legitimate: the steady-state
    solve does not use it. A supplied value of 0.0 is kept, not treated as
    missing.
    """
    for entry in mech.doc.get("siteTypes") or []:
        if entry.get("name", "default") == tname and entry.get("density") is not None:
            return float(entry["density"])
    if len(mech.site_types) == 1:
        for s in mech.surf:
            if "S0" in mech.species[s]:
                return float(mech.species[s]["S0"])
    return None


def build_data(mech):
    """Mechanism -> mechanism data, as a plain dict."""
    surf = mech.surf
    # ions are excluded here: the chemistry sees only the yield channel each
    # ion-driven step declares, which already carries the yield-weighted flux
    gas = [g for g in mech.gas if g not in mech.ions]
    cov_idx = {s: i for i, s in enumerate(surf)}
    gas_idx = {g: i for i, g in enumerate(gas)}

    # A mechanism may have more than one solid: a polymer deposited while a
    # substrate is etched. Each carries its own density, and each reaction says
    # how many atoms of which solid it adds or removes.
    solids = mech.solid
    if not solids:
        raise NotImplementedError("a mechanism needs at least one solid species")
    solid = solids[0]

    # A gas species is traced when it appears on the left of a surface reaction.
    traced = {g: any(g in r["lhs"] for r in mech.reactions) for g in gas}

    reactions = []
    sticking_of = {}  # gas name -> sticking descriptor, from its adsorption step
    fwd = 0           # forward-step counter; a reverse takes its partner's id + "r"
    for j, r in enumerate(mech.reactions, 1):
        # prefactor and temperature exponent; a sticking given as k_ads becomes
        # (4 k_ads / c, -1/2) here, so the consumer needs no kinetic theory.
        # Ea is emitted in eV whatever the file used, so consumers need no
        # unit logic; the source unit is recorded in the conventions.
        pre, beta = mech.rate_constant(r)
        Ea = mech.activation_energy(r)
        if r["type"] == "adsorption":
            const = {"kind": "sticking", "s0": pre, "Ea": Ea, "beta": beta}
        else:
            const = {"kind": "arrhenius", "k0": pre, "Ea": Ea, "beta": beta}

        # The rate exponents. Mass action by default, so an exponent is the
        # reactant's stoichiometric coefficient, but `order:` may state one
        # explicitly. The exponents live here, separately from `nu` below, so a
        # published first-order law over a four-adsorbate step costs no change
        # downstream: the rate uses the order, the balance uses nu.
        gas_factors = [
            {"idx": gas_idx[s], "exp": mech.rate_order(r, s)}
            for s in r["lhs"] if s in gas_idx
        ]
        cov_factors = [
            {"idx": cov_idx[s], "exp": mech.rate_order(r, s)}
            for s in r["lhs"] if s in cov_idx
        ]
        gas_factors = [f for f in gas_factors if f["exp"]]
        cov_factors = [f for f in cov_factors if f["exp"]]
        # reactant free sites consumed, one entry per site type
        free_exp = [mech.rate_order(r, mech.site_token[t])
                    for t in mech.site_types]

        # a reverse step takes its partner's id with an "r" suffix
        if "reverseOf" in r:
            rid = f"R{fwd}r"
        else:
            fwd += 1
            rid = f"R{fwd}"
        # Per-material rate constants. An entry is either a scalar, which
        # scales the prefactor, or a mapping overriding any of the prefactor
        # (`s0`/`k0`), the barrier (`Ea`) and the temperature exponent
        # (`beta`). Both resolve to a full constant here, so the consumer sees
        # one uniform form. "default" covers every material not named.
        base = {"prefactor": pre, "Ea": Ea, "beta": beta}

        def resolve(spec):
            """A material entry -> {prefactor, Ea, beta}."""
            if isinstance(spec, dict):
                out = dict(base)
                if "s0" in spec or "k0" in spec:
                    out["prefactor"] = float(spec.get("s0", spec.get("k0")))
                if "k_ads" in spec:  # same kinetic-theory conversion as the step
                    c = mechanism.speed_coefficient(mech.adsorbing_gas(r))
                    out["prefactor"] = 4.0 * float(spec["k_ads"]) / c
                    out["beta"] = -0.5
                if "Ea" in spec:
                    out["Ea"] = float(spec["Ea"]) * mech.eV_per_energy
                if "beta" in spec:
                    out["beta"] = float(spec["beta"])
                return out
            # a bare number scales the prefactor, leaving the barrier alone
            return {"prefactor": pre * float(spec), "Ea": Ea, "beta": beta}

        materials = dict(r.get("materials") or {})
        has_materials = bool(materials)
        mat_default = resolve(materials.pop("default", 1.0))
        mat_constants = {str(m): resolve(v) for m, v in materials.items()}

        entry = {
            "id": rid,
            "eq": mech.balanced_eq(r),  # free sites written out, as inferred
            "constant": const,
            "gasFactors": sorted(gas_factors, key=lambda f: f["idx"]),
            "covFactors": sorted(cov_factors, key=lambda f: f["idx"]),
            "freeSiteExp": free_exp,
            "nu": [float(mech.net(r, s)) for s in surf],
            # atoms of each solid added per event; negative means removed
            "solidAtoms": [float(mech.net(r, s)) for s in solids],
        }
        if has_materials:
            entry["materialConstants"] = mat_constants
            entry["materialDefault"] = mat_default

        # An ion-driven step carries a yield instead of a rate constant. The
        # particle evaluates Y = A*max(sqrt(E)-sqrt(Eth),0)*f(theta) at each hit
        # and deposits Y*rayWeight, so the rate law sees an ordinary flux.
        if r["type"] in mech.ION_TYPES:
            y = dict(r.get("yield") or {})
            entry["ionYield"] = {
                "ion": mech.ion_of(r),
                "label": f"{rid}_yieldFlux",
                "flux": float(mech.species[mech.ion_of(r)].get("flux", 1.0)),
                # the mean energy travels with the channel so a consumer can
                # evaluate the yield without reaching back to the ion source
                "meanEnergy": float(
                    mech.species[mech.ion_of(r)].get("meanEnergy", 100.0)),
                "A": float(y.get("A", 1.0)),
                "Eth": float(y.get("Eth", 0.0)),
                "B": float(y.get("B", 0.0)),
                "enhanced": r["type"] == "ion_enhanced",
                "materials": {str(m): {"A": float(v["A"]), "Eth": float(v["Eth"])}
                              for m, v in (y.get("materials") or {}).items()},
            }
        if "reverseOf" in r:
            entry["reverseOf"] = f"R{fwd}"
        reactions.append(entry)

        # The particle's sticking comes from the adsorption step that consumes
        # it, on the site type whose free sites that step uses.
        if const["kind"] == "sticking":
            site = next((i for i, e in enumerate(free_exp) if e), 0)
            for f in gas_factors:
                name = gas[f["idx"]]
                descriptor = {
                    "reaction": rid,
                    "s0": const["s0"],
                    "Ea": const["Ea"],
                    "beta": const["beta"],
                    "site": site,
                    "freeSiteExp": free_exp[site],
                }
                # The particle re-emits with the sticking of the material it
                # hit, so it carries the same per-material constants as the
                # adsorption step. Without this the surface solve would be
                # selective while the transport was not.
                if has_materials:
                    descriptor["materialConstants"] = mat_constants
                    descriptor["materialDefault"] = mat_default
                sticking_of[name] = descriptor

    gas_entries = []
    for g in gas:
        d = mech.species[g]
        gas_entries.append(
            {
                "name": g,
                "traced": bool(traced[g]),
                "label": f"{g}_flux" if traced[g] else None,
                "flux": float(d["flux"]) if "flux" in d else None,
                "sticking": sticking_of.get(g),
            }
        )

    # the ion source, when the file declares one
    ion_source = None
    if mech.ions:
        if len(mech.ions) != 1:
            raise NotImplementedError(
                f"expected at most one ion species, got {mech.ions}")
        d = mech.species[mech.ions[0]]
        ion_source = {
            "name": mech.ions[0],
            "flux": float(d["flux"]) if "flux" in d else None,
            "meanEnergy": float(d.get("meanEnergy", 100.0)),
            "sigmaEnergy": float(d.get("sigmaEnergy", 10.0)),
            "exponent": float(d.get("exponent", 200.0)),
            "inflectAngle": float(d.get("inflectAngle", 89.0)),
            "n_l": float(d.get("n_l", 10.0)),
            "minAngle": float(d.get("minAngle", 80.0)),
            "thetaRMin": float(d.get("thetaRMin", 70.0)),
            "thetaRMax": float(d.get("thetaRMax", 90.0)),
        }

    # The solid density. A scalar is one density everywhere; a mapping gives a
    # `default` and a density per material, because a mask and a substrate are
    # not equally dense, so the same removal rate moves them at different speeds.
    def solid_entry(name):
        rho_spec = mech.species[name]["rho"]
        if isinstance(rho_spec, dict):
            if "default" not in rho_spec:
                sys.exit(f"solid '{name}': a rho mapping needs a `default` entry")
            return {
                "name": name,
                "rho": float(rho_spec["default"]),
                "rhoMaterials": {str(m): float(v)
                                 for m, v in rho_spec.items() if m != "default"},
            }
        return {"name": name, "rho": float(rho_spec), "rhoMaterials": {}}

    solid_entries = [solid_entry(s) for s in solids]

    site_index = {t: i for i, t in enumerate(mech.site_types)}
    return {
        "schemaVersion": SCHEMA_VERSION,
        "name": mech.name,
        "siteTypes": [{"name": t, "density": _density_of(mech, t)}
                      for t in mech.site_types],
        "coverages": [{"name": s, "site": site_index[mech.site_of[s]]}
                      for s in surf],
        "gas": gas_entries,
        "solids": solid_entries,
        # the first solid, kept for readers that expect a single one
        "solid": solid_entries[0],
        "ionSource": ion_source,
        "reactions": reactions,
        "constants": {
            "temperature": float(mech.T),
        },
        "conventions": {
            "stickingAppliedIn": "rateLawAndReflection",
            "energyUnit": "eV",  # Ea above is always eV
            "inputEnergyUnit": mech.energy_unit,
            "fluxUnit": "1e15/cm2/s",
            "rhoUnit": "1e22/cm3",
        },
    }


def from_file(path):
    """A reaction file -> mechanism data."""
    return build_data(mechanism.Mechanism(path))


def write(path, data):
    """Write mechanism data where a solver can read it."""
    with open(path, "w") as fh:
        fh.write(json.dumps(data, indent=2) + "\n")
    return path
