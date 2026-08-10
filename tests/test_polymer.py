#!/usr/bin/env python3
"""Two solids in one mechanism: a polymer deposited while silicon is etched.

Until now a mechanism moved ONE solid, so its density was a single number. A
passivating etch moves two at once, with different densities, so every step has
to divide by the density of the solid IT moves:

    v_n = sum_j n_(S,j) r_j / rho_(solid of j)

The competition is a sum of ordinary mass-action terms, not a case switch. What
makes the sidewalls survive is transport, not a rule: they see almost no ion
flux, so their film is never cleared.

Run with the ViennaPS venv:  .venv/bin/python test_polymer.py
"""

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

IR = vc.from_file("polymer_etch.yaml")
MECH = build(IR)


def nodes(dom):
    m = viennals.Mesh()
    ls2.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
    return np.array(m.getNodes())


def main():
    print("Two solids: passivation competing with the etch")

    print("\n1) the reaction file declares two solids, with their own densities")
    names = [(s["name"], s["rho"]) for s in IR["solids"]]
    check("polymer and substrate, each with its density",
          names == [("C", 2.2), ("Si", 5.02)], str(names))
    formed = [r["solidAtoms"] for r in IR["reactions"]]
    check("one step builds the film, others remove film or substrate",
          [1.0, 0.0] in formed and [-1.0, 0.0] in formed
          and [0.0, -1.0] in formed, str(formed))

    print("\n2) front end, data evaluator and C++ agree, on every material")
    for mat in ("Si", "Polymer", "Mask"):
        gamma = vc.mechanism_fluxes(IR, mat)
        th_py, v_py = vc.evaluate(IR, material=mat)
        th_cpp = MECH.solveCoveragesOn(gamma, [0.0] * len(IR["coverages"]), mat)
        v_cpp = MECH.growthRateOn(gamma, th_cpp, mat)
        agree = (max(abs(a - b) for a, b in zip(th_py, th_cpp)) < 1e-12
                 and abs(v_py - v_cpp) < 1e-10)
        check(f"{mat}: coverages and rate agree", agree,
              f"v {v_cpp:+.5f} nm/s   theta {[round(x, 4) for x in th_cpp]}")

    print("\n3) the sign of the velocity flips with the material under the point")
    v = {m: vc.evaluate(IR, material=m)[1] for m in ("Si", "Polymer", "Mask")}
    check("the substrate etches", v["Si"] < 0, f"{v['Si']:+.4f} nm/s")
    check("the film is cleared where the ions strike",
          v["Polymer"] < 0, f"{v['Polymer']:+.4f} nm/s at normal incidence")
    check("with no ion removal the film grows", v["Mask"] > 0,
          f"{v['Mask']:+.4f} nm/s")

    print("\n4) in a trench: the floor etches, the sidewalls keep their film")
    dom = ps.Domain(gridDelta=2.0, xExtent=160.0, yExtent=160.0)
    ps.MakeTrench(domain=dom, trenchWidth=50.0, trenchDepth=30.0,
                  maskHeight=25.0, material=ps.Material.Si,
                  maskMaterial=ps.Material.Mask).apply()
    dom.duplicateTopLevelSet(ps.Material.Polymer)
    before = nodes(dom)

    p = ps.Process(dom, ps.SurfaceChemistry(MECH), 3.0)
    p.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 20
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = 2000; p.setParameters(r)
    p.apply()
    after = nodes(dom)

    def wall(pts, lo, hi):
        band = pts[(pts[:, 1] > lo) & (pts[:, 1] < hi) & (pts[:, 0] > 0)]
        return band[:, 0].min() if len(band) else np.nan

    floor0, floor1 = before[:, 1].min(), after[:, 1].min()
    lo, hi = floor0 + 5.0, floor0 + 20.0
    w0, w1 = wall(before, lo, hi), wall(after, lo, hi)
    print(f"      floor {floor0:.2f} -> {floor1:.2f} nm      "
          f"sidewall x {w0:.2f} -> {w1:.2f} nm")
    check("the floor etches into the substrate", floor1 < floor0 - 5.0,
          f"{floor1 - floor0:+.2f} nm")
    check("the sidewalls gain film instead", w1 < w0 - 2.0,
          f"{w0 - w1:+.2f} nm of film")

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
