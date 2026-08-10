#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lado Filipovic, Institute for Microelectronics, TU Wien
"""Derive a surface-kinetics mechanism from chemical reaction equations.

Input : a reaction file (species + phases, reactions + rate constants).
Output:

  1. atom balance, per reaction; free sites inferred from site conservation
  2. the stoichiometry matrix  nu
  3. the steady-state rate ratios, as the NULL SPACE of the surface stoichiometry
     (this is where  r1 = r2 = r3  comes from -- it is not spotted by eye)
  4. the overall reaction, by contracting the null vector with the gas/solid part
  5. mass-action rate laws, read off the reactions
  6. the coverage balance  dtheta/dt = nu . r = 0, solved symbolically if possible
     and always numerically (Newton with an analytic Jacobian)
  7. the growth rate, from the solid-forming reactions
  8. emitted code: a Python closure and a C++ updateCoverages body

Usage:  python3 -m viennachem --explain silane.yaml
"""

import math
import os
import re
import sys

import sympy as sp
import yaml

KB_EV = 8.617333262e-5  # eV / K
R_GAS = 8.314462618  # J / (mol K)

# Activation energies are canonically eV inside the toolchain. Published tables
# often give kcal/mol against R instead; the two forms are identical, because
# Ea[kcal/mol] / (R T) == Ea[eV] / (kB T).
N_AVOGADRO = 6.02214076e23  # 1 / mol
ELEMENTARY_CHARGE = 1.602176634e-19  # J / eV
KCAL_IN_J = 4184.0  # thermochemical kcal
EV_PER_KCAL_PER_MOL = KCAL_IN_J / (N_AVOGADRO * ELEMENTARY_CHARGE)

ENERGY_UNITS = {
    "eV": 1.0,
    "kcal/mol": EV_PER_KCAL_PER_MOL,
}

# Atomic masses in g/mol, for the mean thermal speed when a sticking
# coefficient is derived from an adsorption rate constant.
ATOMIC_MASS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.95, "K": 39.098, "Ca": 40.078,
    "Ti": 47.867, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
    "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63,
    "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Mo": 95.95,
    "Ru": 101.07, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Ta": 180.95,
    "W": 183.84, "Pt": 195.08, "Au": 196.97, "Pb": 207.2,
}

# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

TOKEN = re.compile(r"^(\d*)\s*(.+)$")
ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_side(side):
    """'2H* + SiH4' -> {'H*': 2, 'SiH4': 1}"""
    out = {}
    for tok in side.split("+"):
        tok = tok.strip()
        if not tok:
            continue
        m = TOKEN.match(tok)
        coeff = int(m.group(1)) if m.group(1) else 1
        name = m.group(2).strip()
        out[name] = out.get(name, 0) + coeff
    return out


def parse_eq(eq):
    """'A -> B' or the reversible 'A <-> B'. Returns (lhs, rhs, reversible)."""
    if "<->" in eq:
        lhs, rhs = eq.split("<->")
        return parse_side(lhs), parse_side(rhs), True
    lhs, rhs = eq.split("->")
    return parse_side(lhs), parse_side(rhs), False


def atoms_of(name):
    """Atom counts. A leading '*' marks a free site (no atoms); a trailing '*'
    marks a surface species, whose atoms come from the rest of the name."""
    if name.startswith("*"):
        return {}
    core = name[:-1] if name.endswith("*") else name
    out = {}
    for el, cnt in ELEMENT.findall(core):
        out[el] = out.get(el, 0) + (int(cnt) if cnt else 1)
    return out


def sites_of(name):
    """Both a free site '*' and an adsorbate 'X*' occupy exactly one site."""
    return 1 if name.endswith("*") else 0


def molar_mass(name):
    """Molar mass in g/mol, summed from the formula."""
    m = 0.0
    for el, n in atoms_of(name).items():
        if el not in ATOMIC_MASS:
            sys.exit(f"no atomic mass known for element '{el}' in '{name}'")
        m += ATOMIC_MASS[el] * n
    return m


def speed_coefficient(name):
    """c in cm/(s sqrt(K)) with mean thermal speed vbar(T) = c sqrt(T).

    Kinetic theory: vbar = sqrt(8 R T / (pi M)).
    """
    M = molar_mass(name) * 1e-3  # kg/mol
    return 100.0 * math.sqrt(8.0 * R_GAS / (math.pi * M))


# --------------------------------------------------------------------------
# mechanism
# --------------------------------------------------------------------------


# The reaction files that ship with the library. They travel inside the package,
# so a bare filename resolves the same way from a checkout, from a wheel, and
# from anywhere on the filesystem. VIENNACHEM_REACTIONS points somewhere else.
REACTIONS_DIR = os.environ.get("VIENNACHEM_REACTIONS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "examples")


def resolve(path):
    """A reaction file, looked for as given and then among the shipped ones."""
    if os.path.exists(path):
        return path
    candidate = os.path.join(REACTIONS_DIR, os.path.basename(path))
    if os.path.exists(candidate):
        return candidate
    sys.exit(f"no reaction file '{path}' here or in {REACTIONS_DIR}")


class Mechanism:
    def __init__(self, path):
        path = resolve(path)
        doc = yaml.safe_load(open(path))
        self.doc = doc
        self.name = doc["name"]
        self.T = float(doc["temperature"])
        self.species = doc["species"]
        self.reactions = doc["surface"]

        self.energy_unit = str(doc.get("energyUnit", "eV"))
        if self.energy_unit not in ENERGY_UNITS:
            sys.exit(f"unknown energyUnit '{self.energy_unit}'; "
                     f"expected one of {sorted(ENERGY_UNITS)}")
        self.eV_per_energy = ENERGY_UNITS[self.energy_unit]

        self.gas = [s for s, d in self.species.items() if d["phase"] == "gas"]
        self.surf = [s for s, d in self.species.items() if d["phase"] == "surface"]
        self.solid = [s for s, d in self.species.items() if d["phase"] == "solid"]
        # Ions carry an energy distribution rather than a flux alone, and their
        # steps are driven by a yield. They are gas species for the purposes of
        # the rate law, because the particle folds the yield into the flux it
        # deposits, so the chemistry sees an ordinary Gamma.
        self.ions = [s for s, d in self.species.items() if d["phase"] == "ion"]
        self.gas = self.gas + self.ions

        # Site types. Each surface species sits on a named site type ("default"
        # if unspecified). Each type has its own free-site pool and its own
        # closure theta_*t = 1 - sum_{i in t} theta_i. With one type the free-site
        # token is "*", so a single-site mechanism is unchanged.
        self.site_of = {s: str(self.species[s].get("site", "default"))
                        for s in self.surf}
        self.site_types = []
        for s in self.surf:
            if self.site_of[s] not in self.site_types:
                self.site_types.append(self.site_of[s])
        if not self.site_types:
            self.site_types = ["default"]
        single = len(self.site_types) == 1
        self.site_token = {t: ("*" if single else "*" + t) for t in self.site_types}
        self.type_of_token = {tok: t for t, tok in self.site_token.items()}

        expanded = []
        for r in self.reactions:
            r["lhs"], r["rhs"], reversible = parse_eq(r["eq"])
            self.infer_free_sites(r)
            expanded.append(r)
            if reversible:
                expanded.append(self.reverse_of(r))
        self.reactions = expanded

    # -- 0. free sites, inferred from site conservation --------------------
    def infer_free_sites(self, r):
        """Fill in empty-site stoichiometry from site conservation, per type.

        Each adsorbate occupies one site of its own type and the total number of
        sites of each type is conserved, so for every type the side with fewer
        adsorbate sites of that type carries the deficit as free sites of that
        type. Free sites written explicitly are dropped and re-derived.
        """
        for tok in self.site_token.values():
            r["lhs"].pop(tok, None)
            r["rhs"].pop(tok, None)
        for t in self.site_types:
            tok = self.site_token[t]
            ls = sum(n for sp_, n in r["lhs"].items() if self.site_of.get(sp_) == t)
            rs = sum(n for sp_, n in r["rhs"].items() if self.site_of.get(sp_) == t)
            if rs > ls:
                r["lhs"][tok] = rs - ls
            elif ls > rs:
                r["rhs"][tok] = ls - rs

    # -- composition -------------------------------------------------------
    def atoms(self, sp_):
        """Atom counts of a species.

        Read from the formula, unless the species declares `atoms:` explicitly.
        A name that is not a formula needs that: the silicon leaving a sputtered
        surface is one Si atom, but `Si` is already the solid, so the product
        has to be called something else and say what it is made of.
        """
        over = self.species.get(sp_, {}).get("atoms")
        if over:
            return {str(el): int(n) for el, n in over.items()}
        return atoms_of(sp_)

    def solid_density(self, solid, material=None):
        """The solid's density, which may be given per material.

        A scalar is one density everywhere. A mapping carries a `default` and a
        density per material, because a mask and a substrate are not equally
        dense, so the same removal rate moves them at different speeds.
        """
        rho = self.species[solid]["rho"]
        if isinstance(rho, dict):
            return float(rho.get(str(material), rho["default"]))
        return float(rho)

    # -- rate orders, when the published law is not mass action -------------
    def rate_order(self, r, sp_):
        """The exponent of `sp_` in the rate law of reaction `r`.

        Mass action is the default: a reactant enters the rate law raised to its
        stoichiometric coefficient. Published surface models are not always
        written that way. A step that consumes four adsorbates may be measured
        to be first order in their coverage, in which case the paper's rate law
        is `k*theta`, not `k*theta^4`.

        `order: {F*: 1}` states that exponent explicitly. The STOICHIOMETRY is
        untouched: the step still consumes four, so nu still counts four, and
        the coverage balance still loses 4r. Only the rate law changes, which is
        possible because the exponents are stored separately from nu.
        """
        n = r["lhs"].get(sp_, 0)
        over = r.get("order")
        if over and sp_ in over:
            return int(over[sp_])
        return n

    def check_orders(self):
        """An explicit order must name something the step actually consumes."""
        for j, r in enumerate(self.reactions, 1):
            for sp_ in (r.get("order") or {}):
                if sp_ not in r["lhs"]:
                    sys.exit(f"[R{j}] order names '{sp_}', which is not a "
                             f"reactant of '{r['eq']}'")

    # -- reversible steps --------------------------------------------------
    def reverse_of(self, r):
        """The reverse of a reversible step, as an ordinary reaction.

        Written 'A <-> B', the reverse is simply 'B -> A': mass action applies to
        the forward step's right-hand side, its stoichiometry is the negation of
        the forward one, and a solid-forming step reverses into etch-back. So a
        reversible step needs no new machinery downstream; it becomes a second
        reaction and the existing solver handles it.

        The reverse of an adsorption is a desorption, so its constant is always a
        thermal rate constant, never a sticking coefficient.
        """
        if "k0_rev" not in r:
            sys.exit(f"reversible step '{r['eq']}' needs a reverse rate constant "
                     f"k0_rev (and Ea_rev)")
        rev = {
            "lhs": dict(r["rhs"]),
            "rhs": dict(r["lhs"]),
            "type": "surface",  # thermal in both directions
            "k0": float(r["k0_rev"]),
            "Ea": float(r.get("Ea_rev", 0.0)),
            "beta": float(r.get("beta_rev", 0.0)),
            "reverseOf": r["eq"],
        }
        # a reverse step is selective on the same materials as its partner
        if "materials" in r:
            rev["materials"] = dict(r["materials"])
        rev["eq"] = self.balanced_eq(rev)
        return rev

    # -- adsorption constant: given, or derived from k_ads -----------------
    def adsorbing_gas(self, r):
        """The gas species consumed by an adsorption step."""
        gs = [s for s in r["lhs"] if s in self.gas]
        if len(gs) != 1:
            sys.exit(f"adsorption step '{r['eq']}' must consume exactly one gas "
                     f"species, found {gs}")
        return gs[0]

    def adsorption_constant(self, r):
        """(prefactor, beta) of the sticking coefficient, s = prefactor T^beta e^(-Ea/kT).

        Supplied as `s0`, this is (s0, 0): a plain Arrhenius sticking.

        Supplied instead as an adsorption rate constant `k_ads` [cm/s], kinetic
        theory converts it: the wall flux is (1/4) n vbar, so k_ads n = s (1/4) n
        vbar gives s = 4 k_ads / vbar. With vbar = c sqrt(T) this is

            s(T) = (4 k_ads0 / c) T^(-1/2) exp(-Ea / kT),

        so the prefactor is 4 k_ads0 / c and beta is -1/2. The molar mass behind
        c comes from the formula, so only k_ads is asked of the user.
        """
        has_s0 = "s0" in r
        has_k = "k_ads" in r
        if has_s0 and has_k:
            sys.exit(f"adsorption step '{r['eq']}' gives both s0 and k_ads; "
                     f"supply one")
        if has_s0:
            return float(r["s0"]), 0.0
        if not has_k:
            sys.exit(f"adsorption step '{r['eq']}' needs either s0 or k_ads")
        c = speed_coefficient(self.adsorbing_gas(r))
        return 4.0 * float(r["k_ads"]) / c, -0.5

    ION_TYPES = ("sputtering", "ion_enhanced")

    def rate_constant(self, r):
        """(prefactor, beta) for any step: adsorption, thermal, or ion driven."""
        if r["type"] == "adsorption":
            return self.adsorption_constant(r)
        if r["type"] in self.ION_TYPES:
            # the yield is folded into the flux the ion particle deposits, so
            # the rate law needs no constant of its own
            return 1.0, 0.0
        return float(r["k0"]), float(r.get("beta", 0.0))

    def ion_of(self, r):
        """The ion species an ion-driven step consumes."""
        ions = [s for s in r["lhs"] if s in self.ions]
        if len(ions) != 1:
            sys.exit(f"ion-driven step '{r['eq']}' must consume exactly one ion "
                     f"species, found {ions}")
        return ions[0]

    def ion_flux(self, r, material=None):
        """The yield-weighted flux an ion-driven step sees at normal incidence.

        Y = A*max(sqrt(E) - sqrt(Eth), 0)*f(theta), and f(theta) = 1 at normal
        incidence for both angular forms, so this is what a blanket surface
        sees. Inside a feature the particle resolves it per ray.
        """
        ion = self.ion_of(r)
        y = dict(r.get("yield") or {})
        A = float(y.get("A", 1.0))
        Eth = float(y.get("Eth", 0.0))
        over = (y.get("materials") or {}).get(str(material)) if material else None
        if over:
            A, Eth = float(over["A"]), float(over["Eth"])
        E = float(self.species[ion].get("meanEnergy", 100.0))
        flux = float(self.species[ion].get("flux", 1.0))
        return A * max(math.sqrt(E) - math.sqrt(Eth), 0.0) * flux

    def activation_energy(self, r):
        """Ea in eV, converted from the file's energyUnit.

        An ion-driven step has no thermal barrier: its threshold is the yield's
        Eth, which the particle applies.
        """
        if r["type"] in self.ION_TYPES:
            return 0.0
        return float(r["Ea"]) * self.eV_per_energy

    def balanced_eq(self, r):
        """The equation with the inferred free sites written out."""
        fmt = lambda d: " + ".join(  # noqa: E731
            f"{'' if n == 1 else n}{sp_}" for sp_, n in d.items()
        )
        return f"{fmt(r['lhs'])} -> {fmt(r['rhs'])}"

    # -- 1. balance check + inferred free sites ----------------------------
    def check_balance(self):
        ok = True
        print("1) atom balance and inferred free sites, per reaction")
        for j, r in enumerate(self.reactions, 1):
            la, ra = {}, {}
            for side, acc in ((r["lhs"], la), (r["rhs"], ra)):
                for sp_, n in side.items():
                    # an ion is a projectile: it delivers energy, not atoms, and
                    # leaves neutralised, so it takes no part in the balance
                    if sp_ in self.ions:
                        continue
                    for el, c in self.atoms(sp_).items():
                        acc[el] = acc.get(el, 0) + n * c
            good = la == ra
            ok &= good
            atoms = ", ".join(f"{e}:{la.get(e,0)}={ra.get(e,0)}" for e in sorted(la | ra))
            parts = []
            for t in self.site_types:
                tok = self.site_token[t]
                if r["lhs"].get(tok):
                    parts.append(f"{r['lhs'][tok]}{tok} left")
                if r["rhs"].get(tok):
                    parts.append(f"{r['rhs'][tok]}{tok} right")
            free = ", ".join(parts) if parts else "none"
            print(f"   [R{j}] {'OK ' if good else 'FAIL'}  atoms {atoms}   free sites: {free}")
            print(f"         inferred:  {self.balanced_eq(r)}")
        if not ok:
            sys.exit("atom balance check failed")
        return ok

    # -- 2. stoichiometry --------------------------------------------------
    def net(self, r, sp_):
        return r["rhs"].get(sp_, 0) - r["lhs"].get(sp_, 0)

    def nu_surface(self):
        """rows = surface species + one free-site pool per type, cols = reactions"""
        rows = self.surf + [self.site_token[t] for t in self.site_types]
        return sp.Matrix([[self.net(r, s) for r in self.reactions] for s in rows]), rows

    # -- 3. steady-state rate ratios = null space --------------------------
    def rate_ratios(self):
        nu, rows = self.nu_surface()
        ns = nu.nullspace()
        if len(ns) != 1:
            return None, nu, rows  # 0 -> over-determined; >1 -> under-determined
        v = ns[0]
        lcm = sp.ilcm(*[sp.Rational(x).q for x in v]) if any(x.q != 1 for x in v) else 1
        v = sp.Matrix([sp.nsimplify(x * lcm) for x in v])
        # igcd needs two arguments; a one-reaction mechanism has a 1-vector
        ints = [int(x) for x in v]
        g = sp.igcd(*ints) if len(ints) > 1 else abs(ints[0]) or 1
        return sp.Matrix([int(x) // g for x in v]), nu, rows

    # -- 4. overall reaction ------------------------------------------------
    def overall(self, ratios):
        net = {}
        for j, r in enumerate(self.reactions):
            for sp_ in self.gas + self.solid:
                net[sp_] = net.get(sp_, 0) + int(ratios[j]) * self.net(r, sp_)
        lhs = {s: -n for s, n in net.items() if n < 0}
        rhs = {s: n for s, n in net.items() if n > 0}
        fmt = lambda d: " + ".join(f"{'' if n==1 else n}{s}" for s, n in d.items())
        return f"{fmt(lhs)} -> {fmt(rhs)}"

    # -- 5. mass-action rate laws ------------------------------------------
    def rate_laws(self):
        """Symbolic rate laws (constants stay symbolic) + their numeric values."""
        th = {s: sp.Symbol(f"theta_{s.rstrip('*')}", nonnegative=True) for s in self.surf}
        th_free = {t: sp.Symbol(f"theta_free_{t}", nonnegative=True)
                   for t in self.site_types}
        # one closure per type: the free fraction is 1 minus the coverages on it
        closure = {
            t: 1 - sum((th[s] for s in self.surf if self.site_of[s] == t),
                       sp.Integer(0))
            for t in self.site_types
        }

        rates, subs = [], {}
        for j, r in enumerate(self.reactions, 1):
            pre, beta = self.rate_constant(r)
            c = sp.Symbol(f"s{j}" if r["type"] == "adsorption" else f"k{j}",
                          positive=True)
            val = (pre * (self.T ** beta)
                   * math.exp(-self.activation_energy(r) / (KB_EV * self.T)))
            subs[c] = val
            r["const"] = c

            expr = c
            for sp_ in r["lhs"]:
                # mass action unless the reaction states an explicit order
                n = self.rate_order(r, sp_)
                if sp_ in self.surf:
                    expr *= th[sp_] ** n
                elif sp_ in self.type_of_token:
                    expr *= th_free[self.type_of_token[sp_]] ** n
                elif sp_ in self.ions:
                    # The ion particle deposits a YIELD-WEIGHTED flux, so the
                    # rate law sees GY_j = Y_j * Gamma_ion rather than the bare
                    # ion flux, and each step has its own yield. At normal
                    # incidence f(theta) = 1, which is the blanket value.
                    GY = sp.Symbol(f"GY_{sp_}_{j}", positive=True)
                    subs[GY] = self.ion_flux(r)
                    expr *= GY ** n
                elif sp_ in self.gas:
                    G = sp.Symbol(f"Gamma_{sp_}", positive=True)
                    subs[G] = float(self.species[sp_]["flux"])
                    expr *= G ** n
            rates.append(expr)
        return rates, th, th_free, closure, subs

    # -- 6/7. steady state + growth rate ------------------------------------
    def solve(self):
        rates, th, th_free, closure, subs = self.rate_laws()

        # eliminate each type's free-site fraction via its own closure
        elim = {th_free[t]: closure[t] for t in self.site_types}
        eqs = [
            sum(self.net(r, s) * rates[j] for j, r in enumerate(self.reactions)).subs(elim)
            for s in self.surf  # the site rows are implied by the closures
        ]
        unknowns = [th[s] for s in self.surf]

        F = sp.Matrix(eqs)
        J = F.jacobian(unknowns)
        n = len(unknowns)
        Fn = F.subs(subs)
        Jn = J.subs(subs)
        f = sp.lambdify(unknowns, [Fn[i] for i in range(n)], "math")
        jac = sp.lambdify(
            unknowns, [[Jn[i, j] for j in range(n)] for i in range(n)], "math"
        )

        # Newton from the bare surface, clamped to the physical simplex
        x = [1e-6] * n
        for _ in range(200):
            Fx = sp.Matrix(f(*x))
            Jx = sp.Matrix(jac(*x))
            dx = Jx.solve(-Fx)
            x = [max(0.0, min(1.0, float(x[i] + dx[i]))) for i in range(n)]
            if max(abs(float(d)) for d in dx) < 1e-14:
                break
        theta = dict(zip(self.surf, x))

        # every solid contributes its own atom count over its own density, so
        # a mechanism depositing one solid while etching another adds up right
        growth = sum(
            self.net(r, s) * rates[j] / self.solid_density(s)
            for s in self.solid
            for j, r in enumerate(self.reactions)
        )
        num = {th[s]: theta[s] for s in self.surf}
        num.update(subs)
        for t in self.site_types:
            num[th_free[t]] = 1 - sum(theta[s] for s in self.surf
                                      if self.site_of[s] == t)
        v = float(growth.subs(num))
        return dict(
            theta=theta, v=v, rates=rates, th=th, th_free=th_free, closure=closure,
            growth=growth, subs=subs, F=F, J=J, unknowns=unknowns, num=num,
        )

    # -- 8. symbolic relations implied by the null space --------------------
    def null_relations(self, ratios, rates, th, th_free):
        """For each pair with equal null-space weight, r_i = r_j is an exact
        relation. Solve it symbolically -- this is where sqrt() appears.

        There is nothing to say when the null space is not one dimensional: the
        rates are then not pinned to each other, and the Newton solve is the
        only route to the steady state."""
        out = []
        if ratios is None:
            return out
        for i in range(len(rates)):
            for j in range(i + 1, len(rates)):
                if int(ratios[i]) != int(ratios[j]):
                    continue
                eq = sp.Eq(rates[i], rates[j])
                for sym in list(th.values()):
                    try:
                        sols = sp.solve(eq, sym, simplify=True)
                    except Exception:
                        continue
                    for so in sols:
                        if so.is_real is False:
                            continue
                        if any(f in so.free_symbols for f in th_free.values()):
                            out.append((i + 1, j + 1, sym, sp.simplify(so)))
        return out


def explain(path):
    m = Mechanism(path)
    print(f"mechanism: {m.name}   T = {m.T} K\n")
    m.check_balance()
    m.check_orders()

    ratios, nu, rows = m.rate_ratios()
    print("\n2) surface stoichiometry matrix nu (rows = surface species + free site)")
    for i, s in enumerate(rows):
        print(f"   {s:>7}: {[int(nu[i, j]) for j in range(nu.cols)]}")

    print("\n3) steady-state rate ratios = null space of nu")
    if ratios is None:
        print("   null space is not 1-D -> no unique ratio; use the full Newton solve")
    else:
        print("   " + " = ".join(f"r{j+1}/{int(ratios[j])}" for j in range(len(ratios))))
        print(f"   i.e.  {' = '.join(f'r{j+1}' for j in range(len(ratios)))}"
              if all(int(x) == 1 for x in ratios) else "")

        print("\n4) overall reaction (null vector contracted with gas/solid stoichiometry)")
        print(f"   {m.overall(ratios)}")

    res = m.solve()
    theta, rates, th, th_free = res["theta"], res["rates"], res["th"], res["th_free"]

    print("\n5) mass-action rate laws (read off the reactions, not supplied)")
    for j, r in enumerate(rates, 1):
        print(f"   r{j} = {r}")
    for t in m.site_types:
        print(f"   with  {th_free[t]} = {res['closure'][t]}")

    print("\n6) symbolic relations implied by the null space (r_i = r_j)")
    rels = m.null_relations(ratios, rates, th, th_free)
    seen = set()
    for i, j, sym, so in rels:
        key = (sym, sp.srepr(so))
        if key in seen:
            continue
        seen.add(key)
        print(f"   from r{i} = r{j}:   {sym} = {so}")

    print("\n7) coverage balance  dtheta/dt = nu . r = 0  (Newton, analytic Jacobian)")
    for s, val in theta.items():
        print(f"   theta_{s.rstrip('*'):<5} = {val:.6e}")
    for t in m.site_types:
        free = 1 - sum(theta[s] for s in m.surf if m.site_of[s] == t)
        print(f"   theta_free[{t}]  = {free:.6e}")
    rj = [float(r.subs(res["num"])) for r in rates]
    ratio_note = (f"  (null ratios {[int(x) for x in ratios]})"
                  if ratios is not None else "")
    print(f"   rates at solution: {['%.6e' % x for x in rj]}{ratio_note}")

    print("\n8) growth rate from the solid-forming reactions")
    print(f"   v_n = {res['growth']}")
    print(f"       = {res['v']:.6f} nm/s")

    print("\n9) emitted C++ (updateCoverages residual + Jacobian, SymPy ccode)")
    F, J, unknowns = res["F"], res["J"], res["unknowns"]
    for i in range(F.rows):
        print(f"   F[{i}] = {sp.ccode(sp.expand(F[i]))};")
    for i in range(J.rows):
        for j in range(J.cols):
            print(f"   J[{i}][{j}] = {sp.ccode(sp.expand(J[i, j]))};")
