#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lado Filipovic, Institute for Microelectronics, TU Wien
"""Evaluate mechanism data: coverages and growth rate at a given local flux.

This module deliberately uses only `math` and `json`: no SymPy, no YAML, no
ViennaPS. That is the point. If it reproduces the front end's answer from the
mechanism data alone, the data is complete, and it doubles as the reference any
other implementation is checked against.

Everything here mirrors what the C++ will do per surface point:

  rate_j    = k_j * prod_g Gamma_g^n_gj * theta_free^n_*j * prod_i theta_i^n_ij
  F_i       = sum_j nu_ij * rate_j                             (= 0 at steady state)
  J_ik      = sum_j nu_ij * d(rate_j)/d(theta_k)
  v_n       = (1/rho) * sum_j solidAtoms_j * rate_j

The Jacobian is built from the stored exponents by the product rule, including
the chain term from theta_free = 1 - sum_i theta_i. Derivatives are evaluated by
multiplying the remaining factors rather than dividing, so a coverage of zero is
safe.
"""

import json
import math

KB_EV = 8.617333262e-5  # eV / K


# --------------------------------------------------------------------------
# rate constants
# --------------------------------------------------------------------------


def _arrhenius(pref, Ea, beta, T):
    """pref * T^beta * exp(-Ea / kT); beta = 0 is the plain Arrhenius form."""
    k = pref * math.exp(-Ea / (KB_EV * T))
    return k * (T ** beta) if beta else k


def constant_for(rx, material):
    """The (prefactor, Ea, beta) this reaction takes on `material`.

    With no material context, or no per-material block, the reaction's own
    constant applies.
    """
    c = rx["constant"]
    own = (c["s0"] if c["kind"] == "sticking" else c["k0"],
           c["Ea"], c.get("beta", 0.0))
    if material is None or "materialConstants" not in rx:
        return own
    spec = rx["materialConstants"].get(material) or rx.get("materialDefault")
    if spec is None:
        return own
    return spec["prefactor"], spec["Ea"], spec.get("beta", 0.0)


def rate_constants(ir, temperature=None, material=None):
    """Evaluate every reaction's constant at T, for `material`."""
    T = temperature if temperature is not None else ir["constants"]["temperature"]
    out = []
    for rx in ir["reactions"]:
        pref, Ea, beta = constant_for(rx, material)
        out.append(_arrhenius(pref, Ea, beta, T))
    return out


def sticking(ir, gas_name, temperature=None):
    """Effective sticking prefactor s(T) and its free-site exponent."""
    T = temperature if temperature is not None else ir["constants"]["temperature"]
    for g in ir["gas"]:
        if g["name"] == gas_name:
            st = g["sticking"]
            if st is None:
                return None
            return (
                _arrhenius(st["s0"], st["Ea"], st.get("beta", 0.0), T),
                st["freeSiteExp"],
            )
    raise KeyError(gas_name)


def default_fluxes(ir):
    """Incident flux per gas species, from the reaction file."""
    return [(g["flux"] or 0.0) for g in ir["gas"]]


def ion_channel_flux(rx, material=None):
    """The yield-weighted flux an ion-driven step sees at NORMAL incidence.

    The ion particle evaluates Y = A*max(sqrt(E)-sqrt(Eth),0)*f(theta) per ray
    and deposits Y*rayWeight, so the chemistry reads a flux rather than a yield.
    At normal incidence f = 1 for both angular forms, and the mean energy stands
    in for the distribution. That is what a blanket surface sees, so it is the
    analytic reference a simulation on a flat surface must reproduce.
    """
    y = rx.get("ionYield")
    if y is None:
        return None
    A, Eth = y["A"], y["Eth"]
    if material is not None:
        over = (y.get("materials") or {}).get(str(material))
        if over is not None:
            A, Eth = over["A"], over["Eth"]
    return A * max(math.sqrt(y["meanEnergy"]) - math.sqrt(Eth), 0.0) * y["flux"]


def mechanism_fluxes(ir, material=None):
    """The gamma vector a built ViennaPS mechanism expects.

    A reader declares one yield channel per ion-driven reaction AFTER the gas
    species, in reaction order, so the vector is the gas fluxes followed by the
    channels' normal-incidence fluxes.
    """
    gamma = default_fluxes(ir)
    for rx in ir["reactions"]:
        f = ion_channel_flux(rx, material)
        if f is not None:
            gamma.append(f)
    return gamma


# --------------------------------------------------------------------------
# rates and derivatives
# --------------------------------------------------------------------------


def coverage_sites(ir):
    """Site-type index of each coverage, in coverage order."""
    return [c["site"] for c in ir["coverages"]]


def free_fractions(ir, theta):
    """The free-site fraction of each site type: theta_*t = 1 - sum_{i in t} theta_i."""
    nsite = len(ir["siteTypes"])
    site = coverage_sites(ir)
    occ = [0.0] * nsite
    for i, th in enumerate(theta):
        occ[site[i]] += th
    return [max(1.0 - o, 0.0) for o in occ]


def _gas_part(rx, k, gammas, material=None):
    val = k
    for f in rx["gasFactors"]:
        val *= gammas[f["idx"]] ** f["exp"]
    # an ion-driven step reads its yield channel's flux, which the ion particle
    # produces; at normal incidence that flux has a closed form
    ion = ion_channel_flux(rx, material)
    if ion is not None:
        val *= ion
    return val


def _free_part(rx, frees):
    """Product of theta_*t^n_t over all site types."""
    val = 1.0
    for t, n in enumerate(rx["freeSiteExp"]):
        if n:
            val *= frees[t] ** n
    return val


def blocked_fraction(ir, theta):
    """The fraction of the surface a blocking pool leaves free.

    The pool is declared with a weight per blocker and a capacity. Adduct
    blockers count in proportion to the free fraction itself, which makes the
    definition implicit; it is linear, so it is solved directly:

        t = a + b(1 - t)   =>   t = (a + b) / (1 + b)
    """
    blk = ir.get("blocking")
    if not blk:
        return 1.0
    cap = blk.get("capacity", 1.0) or 1.0
    a = sum(w["weight"] * theta[w["idx"]] for w in blk.get("weights") or []) / cap
    b = sum(w["weight"] * theta[w["idx"]] for w in blk.get("adducts") or []) / cap
    t = (a + b) / (1.0 + b)
    return max(0.0, min(1.0, 1.0 - t))


def rate(rx, k, gammas, theta, frees, material=None, free_fraction=None):
    val = _gas_part(rx, k, gammas, material) * _free_part(rx, frees)
    for f in rx["covFactors"]:
        val *= theta[f["idx"]] ** f["exp"]
    if free_fraction is not None and rx.get("blocked"):
        val *= free_fraction
    return val


def drate_dtheta(rx, k, gammas, theta, frees, kidx, site, material=None):
    """d(rate)/d(theta_kidx) by the product rule over the stored exponents.

    theta_kidx affects the rate twice: directly as a coverage factor, and
    through the free fraction of its own site type (site[kidx]), since
    theta_*t = 1 - sum_{i in t} theta_i.
    """
    base = _gas_part(rx, k, gammas, material)
    total = 0.0

    # theta_kidx appearing explicitly as a coverage factor; the free part is intact
    for f in rx["covFactors"]:
        if f["idx"] != kidx:
            continue
        e = f["exp"]
        term = base * e * theta[kidx] ** (e - 1)
        for g in rx["covFactors"]:
            if g is not f:
                term *= theta[g["idx"]] ** g["exp"]
        term *= _free_part(rx, frees)
        total += term

    # through theta_*t of kidx's own type t; d(theta_*t)/d(theta_kidx) = -1
    tk = site[kidx]
    n = rx["freeSiteExp"][tk]
    if n:
        term = -base * n * frees[tk] ** (n - 1)
        for g in rx["covFactors"]:
            term *= theta[g["idx"]] ** g["exp"]
        for t, m in enumerate(rx["freeSiteExp"]):   # the other types' free factors
            if t != tk and m:
                term *= frees[t] ** m
        total += term

    return total


# --------------------------------------------------------------------------
# dense linear solve (small systems; mirrors what the C++ will do)
# --------------------------------------------------------------------------


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            return None
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(c + 1, n):
            fac = M[r][c] / piv
            if fac:
                for cc in range(c, n + 1):
                    M[r][cc] -= fac * M[c][cc]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


# --------------------------------------------------------------------------
# transient step, for a cyclic process
# --------------------------------------------------------------------------


def step_coverages(ir, theta, gammas, dt, max_change=2e-4, ks=None,
                   material=None, max_sub_steps=4000000, grown=None,
                   free_fraction=None, extents=None, trace=None, t0=0.0):
    """Integrate the coverages over dt, and return them with what was grown.

    The same rule the C++ model applies, so a figure drawn from here and a run
    of the driver describe one model. A cyclic process integrates thousands of
    pulses in sequence, so a rule that is merely convergent is not enough: an
    error that does not cancel between the states a reaction consumes and the
    states it produces accumulates over the cycles. Every reaction here
    transfers occupancy from the states it consumes into the states it
    produces, written as a rate coefficient q[i][k], and the sub-step solves

        (I + h diag(L) - h q) theta(t+h) = theta(t),   L[k] = sum_i q[i][k]

    The columns of that matrix sum to one, so the occupancy of every site type
    is conserved exactly whatever the sub-step, and the matrix is an M-matrix,
    so the result stays non-negative. The free fraction of each site type is
    carried as an extra unknown, so a reaction that takes a free site or that
    frees one conserves as well.

    `free_fraction`, where a mechanism states steric blocking, replaces the
    free-site part of every rate law. It follows the coverages, so pass the
    callable `blocked_fraction` rather than a value and it is re-evaluated on
    every sub-step. `extents`, a list one entry per reaction,
    accumulates the time integral of each reaction rate over dt, which is what
    a per-channel accounting of the growth needs. `trace`, a list, receives
    `(t, theta, grown)` after every sub-step, which is what a figure of the
    coverages through a pulse reads.
    """
    if ks is None:
        ks = rate_constants(ir, material=material)
    site = coverage_sites(ir)
    nsite = len(ir["siteTypes"])
    dens = [t.get("density") for t in ir["siteTypes"]]
    scale = [1e15 / dens[site[i]] if dens[site[i]] else 1.0
             for i in range(len(site))]
    n = len(theta)
    N = n + nsite
    theta = list(theta)
    total = 0.0 if grown is None else grown
    elapsed, sub = 0.0, 0
    while elapsed < dt - 1e-12 and sub < max_sub_steps:
        sub += 1
        free = free_fractions(ir, theta)
        blocked = (free_fraction(ir, theta) if callable(free_fraction)
                   else free_fraction)
        x = theta + free
        q = [[0.0] * N for _ in range(N)]
        rj_all = [0.0] * len(ir["reactions"])
        for j, rx in enumerate(ir["reactions"]):
            rj = rate(rx, ks[j], gammas, theta, free, material=material,
                      free_fraction=blocked)
            rj_all[j] = rj
            if rj == 0.0:
                continue
            # one site type at a time: each is conserved on its own, and a
            # reaction touching two types moves each separately
            for t in range(nsite):
                gain = [0.0] * N
                lose = [0.0] * N
                gv = lv = 0.0
                for i in range(n):
                    if site[i] != t or rx["nu"][i] == 0.0:
                        continue
                    w = rx["nu"][i] * scale[i]
                    if w > 0.0:
                        gain[i] = w; gv += w
                    else:
                        lose[i] = -w; lv += -w
                # what the tracked states do not balance comes from, or returns
                # to, the free fraction of the type
                if gv > lv:
                    lose[n + t] += gv - lv; lv = gv
                elif lv > gv:
                    gain[n + t] += lv - gv; gv = lv
                if gv <= 0.0:
                    continue
                w = rj / gv
                for c in range(N):
                    if lose[c] == 0.0:
                        continue
                    col = w * lose[c] / max(x[c], 1e-300)
                    for i in range(N):
                        if gain[i]:
                            q[i][c] += gain[i] * col
        L = [sum(q[i][c] for i in range(N)) for c in range(N)]
        fastest = 0.0
        for i in range(n):
            net = sum(q[i][c] * x[c] for c in range(N)) - L[i] * x[i]
            fastest = max(fastest, abs(net))
        h = dt - elapsed
        if fastest > 0.0:
            h = min(h, max_change / fastest)
        total += growth_rate(ir, theta, gammas, material=material) * h
        if extents is not None:
            for j in range(len(rj_all)):
                extents[j] += rj_all[j] * h
        A = [[-h * q[i][c] for c in range(N)] for i in range(N)]
        for i in range(N):
            A[i][i] += 1.0 + h * L[i]
        y = _solve(A, x)
        if y is not None:
            theta = [min(1.0, max(0.0, v)) for v in y[:n]]
        for t in range(nsite):
            idx = [i for i in range(n) if site[i] == t]
            occ = sum(theta[i] for i in idx)
            if occ > 1.0:
                for i in idx:
                    theta[i] /= occ
        elapsed += h
        if trace is not None:
            trace.append((t0 + elapsed, list(theta), total))
    return theta, total


# --------------------------------------------------------------------------
# steady-state solve
# --------------------------------------------------------------------------


def solve_coverages(
    ir, gammas=None, temperature=None, theta0=None, tol=1e-14, max_iter=200,
    material=None
):
    """Newton solve of sum_j nu_ij r_j = 0, clamped to the physical simplex."""
    ks = rate_constants(ir, temperature, material)
    if gammas is None:
        gammas = default_fluxes(ir)
    ncov = len(ir["coverages"])
    if ncov == 0:
        return []          # nothing adsorbs: pure sputtering has no balance
    nsite = len(ir["siteTypes"])
    site = coverage_sites(ir)
    rxs = ir["reactions"]

    theta = list(theta0) if theta0 is not None else [1e-8] * ncov

    for _ in range(max_iter):
        frees = free_fractions(ir, theta)

        F = [0.0] * ncov
        J = [[0.0] * ncov for _ in range(ncov)]
        for j, rx in enumerate(rxs):
            rj = rate(rx, ks[j], gammas, theta, frees, material)
            for i in range(ncov):
                nu = rx["nu"][i]
                if nu:
                    F[i] += nu * rj
            for kidx in range(ncov):
                d = drate_dtheta(rx, ks[j], gammas, theta, frees, kidx, site,
                                 material)
                if d:
                    for i in range(ncov):
                        nu = rx["nu"][i]
                        if nu:
                            J[i][kidx] += nu * d

        # A coverage that no reaction touches on this material is
        # unconstrained: its Jacobian row is empty and its residual is zero, so
        # the system is singular. Nothing produces it, so pin it to zero. This
        # is what a per-material block does when it switches off every step
        # that feeds a species.
        for i in range(ncov):
            if abs(F[i]) < 1e-30 and all(abs(J[i][c]) < 1e-30 for c in range(ncov)):
                J[i][i] = 1.0
                F[i] = theta[i]

        dx = _solve(J, [-f for f in F])
        if dx is None:
            break

        # damp so the step keeps each theta_i >= 0 and each type's sum <= 1
        # A coverage already at zero has no room to give, so it must not
        # scale the step to nothing; the clamp below holds it at zero.
        scale = 1.0
        for i in range(ncov):
            if theta[i] > 0.0 and theta[i] + dx[i] < 0.0 and dx[i] < 0.0:
                scale = min(scale, 0.5 * theta[i] / (-dx[i]))
        for t in range(nsite):
            cur = sum(theta[i] for i in range(ncov) if site[i] == t)
            step = sum(dx[i] for i in range(ncov) if site[i] == t)
            if cur + scale * step > 1.0 and step > 0:
                scale = min(scale, max(0.0, 0.5 * (1.0 - cur) / step))

        new = [
            min(1.0, max(0.0, theta[i] + scale * dx[i])) for i in range(ncov)
        ]
        delta = max(abs(new[i] - theta[i]) for i in range(ncov))
        theta = new
        if delta < tol:
            break

    return theta


def growth_rate(ir, theta, gammas=None, temperature=None, material=None):
    """v_n in nm/s, with flux in 1e15/cm2/s and rho in 1e22/cm3."""
    ks = rate_constants(ir, temperature, material)
    if gammas is None:
        gammas = default_fluxes(ir)
    frees = free_fractions(ir, theta)
    rho = [solid_density(ir, s, material) for s in range(len(ir["solids"]))]
    total = 0.0
    for j, rx in enumerate(ir["reactions"]):
        if any(rx["solidAtoms"]):
            r = rate(rx, ks[j], gammas, theta, frees, material)
            # each solid moves the surface by its own atom count over its own
            # density, so deposition of one and removal of another add up
            for s, n in enumerate(rx["solidAtoms"]):
                if n:
                    total += n * r / rho[s]
    return total


def solid_density(ir, solid=0, material=None):
    """The density of solid `solid` under a point, which may differ per material."""
    entry = ir["solids"][solid]
    per = entry.get("rhoMaterials") or {}
    if material is not None and str(material) in per:
        return per[str(material)]
    return entry["rho"]


def evaluate(ir, gammas=None, temperature=None, theta0=None, material=None):
    theta = solve_coverages(ir, gammas, temperature, theta0, material=material)
    v = growth_rate(ir, theta, gammas, temperature, material)
    return theta, v


# --------------------------------------------------------------------------


def main():
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "silane.mechanism.json"
    ir = json.load(open(path))
    theta, v = evaluate(ir)
    print(f"mechanism: {ir['name']}   T = {ir['constants']['temperature']} K")
    for c, t in zip(ir["coverages"], theta):
        print(f"   theta_{c['name'].rstrip('*'):<6} = {t:.6e}")
    for st, fr in zip(ir["siteTypes"], free_fractions(ir, theta)):
        print(f"   theta_free[{st['name']}] = {fr:.6e}")
    print(f"   v_n          = {v:.6f} nm/s")
    for g in ir["gas"]:
        if g["traced"]:
            s, n = sticking(ir, g["name"])
            print(f"   particle {g['name']}: s(T) = {s:.4e}, s_eff = s * theta_free^{n}")


if __name__ == "__main__":
    main()
