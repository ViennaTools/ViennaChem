#!/usr/bin/env python3
"""Ion gate: an ion-driven step etches, with the yield the particle computes.

An ion yield

    Y = A * max(sqrt(E) - sqrt(Eth), 0) * f(theta)

depends on the ion energy distribution and the local incidence angle, both
resolved per ray. It cannot be written as k*Gamma^n*theta^m. It fits anyway
because the ION PARTICLE evaluates the yield at every hit and deposits a
YIELD-WEIGHTED flux, so the surface chemistry sees an ordinary flux and stays
mass action. Nothing in the solver knows about ions.

Etching comes from the stoichiometry: the solid sits on the left of the
reaction, so `solidAtoms` is negative and the surface velocity is negative.

Run with the ViennaPS venv:  .venv/bin/python test_ion.py
"""

import copy
import math
import sys

import numpy as np
import viennaps as ps
import viennals
import viennals.d2 as ls2

import viennachem as vc
from support import build

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


ps.setDimension(2)
ps.setNumThreads(16)
ps.Length.setUnit("nm")
ps.Time.setUnit("s")
ps.Logger.setLogLevel(ps.LogLevel.WARNING)

IR = vc.from_file("ar_sputter.yaml")
MECH = build(IR)

E, GAMMA, RHO = 100.0, 50.0, 5.02


def analytic(A, Eth):
    """Normal incidence: f(theta) = 1, so the rate is -Y*Gamma/rho."""
    return -A * (math.sqrt(E) - math.sqrt(Eth)) * GAMMA / RHO


def top(dom):
    m = viennals.Mesh()
    ls2.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
    return np.array(m.getNodes())[:, 1].max()


def blanket(material, t=100.0, rays=3000):
    dom = ps.Domain(gridDelta=2.0, xExtent=100.0, yExtent=100.0)
    ps.MakePlane(dom, 0.0, material).apply()
    y0 = top(dom)
    p = ps.Process(dom, ps.SurfaceChemistry(MECH), t)
    p.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 20
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    return (top(dom) - y0) / t


def without(ir, which):
    """A copy with either the neutral flux or the ion flux switched off."""
    out = copy.deepcopy(ir)
    if which == "F":
        for g in out["gas"]:
            if g["name"] == "F":
                g["flux"] = 0.0
    else:
        for r in out["reactions"]:
            if r.get("ionYield"):
                r["ionYield"]["flux"] = 0.0
    return out


def blanket_with(mech, material, t=100.0, rays=3000):
    dom = ps.Domain(gridDelta=2.0, xExtent=100.0, yExtent=100.0)
    ps.MakePlane(dom, 0.0, material).apply()
    y0 = top(dom)
    p = ps.Process(dom, ps.SurfaceChemistry(mech), t)
    p.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 30
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    return (top(dom) - y0) / t


def main():
    print("Ion gate: physical sputtering")

    print("\n1) the reaction file declares an ion and a yield")
    check("an ion source is present", IR["ionSource"] is not None,
          f"meanEnergy={IR['ionSource']['meanEnergy']}")
    r1 = IR["reactions"][0]
    check("the step carries a yield, not a rate constant",
          "ionYield" in r1 and r1["ionYield"]["A"] == 0.02,
          str(r1.get("ionYield", {}).get("A")))
    check("the ion is not also an ordinary gas factor",
          r1["gasFactors"] == [], str(r1["gasFactors"]))
    check("the solid is consumed, so the step etches",
          r1["solidAtoms"] == [-1.0], str(r1["solidAtoms"]))

    print("\n2) the etch rate matches the yield at normal incidence")
    r_si = blanket(ps.Material.Si)
    a_si = analytic(0.02, 20.0)
    check("Si etches", r_si < 0, f"{r_si:.5e} nm/s")
    check("Si rate matches the analytic yield",
          abs(r_si - a_si) / abs(a_si) < 0.05,
          f"{r_si:.5e} vs {a_si:.5e} ({100*abs(r_si-a_si)/abs(a_si):.2f}%)")

    print("\n3) the per-material yield gives mask selectivity")
    r_mask = blanket(ps.Material.Mask)
    a_mask = analytic(0.002, 40.0)
    check("Mask rate matches its own yield",
          abs(r_mask - a_mask) / abs(a_mask) < 0.05,
          f"{r_mask:.5e} vs {a_mask:.5e}")
    sel, sel_expect = r_si / r_mask, a_si / a_mask
    check("selectivity matches the yield ratio",
          abs(sel - sel_expect) / sel_expect < 0.05,
          f"{sel:.2f} vs {sel_expect:.2f}")

    print("\n4) ion-enhanced etching: the yield multiplies a coverage")
    ir_e = vc.from_file("cf4ar_etch.yaml")
    mech_e = build(ir_e)
    r2 = ir_e["reactions"][1]
    check("the enhanced step carries a coverage factor",
          r2["covFactors"] == [{"idx": 0, "exp": 4}], str(r2["covFactors"]))
    check("its free sites were inferred on the product side",
          r2["eq"].endswith("+ 4*"), r2["eq"])
    check("the ion takes no part in the atom balance",
          r2["ionYield"]["enhanced"] is True)

    # analytic steady state at normal incidence, where f_ie = 1
    E, Gion, GF, rho = 100.0, 50.0, 5000.0, 5.02
    Y_ie = 0.15 * (math.sqrt(E) - math.sqrt(10.0))
    Y_sp = 0.01 * (math.sqrt(E) - math.sqrt(25.0))
    lo, hi = 0.0, 1.0
    for _ in range(200):                      # F balance: s*GF*(1-th) = 4*Y_ie*Gion*th^4
        th = 0.5 * (lo + hi)
        if 0.3 * GF * (1.0 - th) > 4.0 * Y_ie * Gion * th ** 4:
            lo = th
        else:
            hi = th
    th = 0.5 * (lo + hi)
    v_analytic = -(Y_ie * Gion * th ** 4 + Y_sp * Gion) / rho

    v_sim = blanket_with(mech_e, ps.Material.Si, t=50.0)
    check("the etch rate matches the analytic steady state",
          abs(v_sim - v_analytic) / abs(v_analytic) < 0.05,
          f"{v_sim:.5f} vs {v_analytic:.5f} "
          f"({100*abs(v_sim-v_analytic)/abs(v_analytic):.2f}%)")

    print("\n5) ion and neutral together etch far faster than either alone")
    ions_only = blanket_with(build(without(ir_e, "F")),
                             ps.Material.Si, t=50.0)
    neut_only = blanket_with(build(without(ir_e, "ion")),
                             ps.Material.Si, t=50.0)
    print(f"      ions only {ions_only:.4e}   neutrals only {neut_only:.4e}"
          f"   both {v_sim:.4e} nm/s")
    check("fluorine alone does not etch", abs(neut_only) < 1e-6,
          f"{neut_only:.3e}")
    check("the combination beats the sum of the parts",
          abs(v_sim) > 5.0 * abs(ions_only + neut_only),
          f"{abs(v_sim)/abs(ions_only+neut_only):.1f}x")

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
