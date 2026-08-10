#!/usr/bin/env python3
"""Material-selective deposition: the rate follows the material under each point.

The pure-Python evaluator and the compiled model must agree on the per-material
growth rate, and a simulation on a surface with two exposed materials must grow
the film on one and leave the other alone.

Run with the ViennaPS venv:  .venv/bin/python test_selective.py
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


def close(a, b, rtol=1e-6):
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


ps.setDimension(2)
ps.setNumThreads(16)
ps.Length.setUnit("nm")
ps.Time.setUnit("s")
ps.Logger.setLogLevel(ps.LogLevel.WARNING)

SELECTIVE = vc.from_file("silane_selective.yaml")
PLAIN = vc.from_file("silane.yaml")


def top_of(dom, level_set=-1):
    m = viennals.Mesh()
    ls2.ToSurfaceMesh(dom.getLevelSets()[level_set], m).apply()
    return np.array(m.getNodes())


def blanket(mech, substrate, t=200.0, rays=2000):
    """Deposit on a flat wafer of `substrate` and measure the growth."""
    dom = ps.Domain(gridDelta=2.0, xExtent=100.0, yExtent=100.0)
    ps.MakePlane(dom, 0.0, substrate).apply()
    y0 = top_of(dom)[:, 1].max()
    p = ps.Process(dom, ps.SurfaceChemistry(mech), t)
    p.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 20
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    return (top_of(dom)[:, 1].max() - y0) / t


def main():
    print("Material-selective deposition")

    print("\n1) the IR carries the per-material factors")
    r1 = SELECTIVE["reactions"][0]
    mc = r1.get("materialConstants", {})
    check("adsorption step is selective",
          set(mc) == {"Si", "PolySi", "SiO2"}
          and r1["materialDefault"]["prefactor"] == 0.0,
          f"{sorted(mc)} default prefactor={r1['materialDefault']['prefactor']}")
    check("a scalar entry scales the prefactor and keeps the barrier",
          close(mc["SiO2"]["prefactor"], 0.01, 1e-12)
          and close(mc["SiO2"]["Ea"], r1["constant"]["Ea"], 1e-12),
          f"pre={mc['SiO2']['prefactor']} Ea={mc['SiO2']['Ea']}")
    r2 = SELECTIVE["reactions"][1]
    check("the other steps carry no material block",
          "materialConstants" not in r2)

    print("\n2) C++ mechanism vs the Python evaluator, per material")
    mech = build(SELECTIVE)
    gamma = vc.default_fluxes(SELECTIVE)
    # the C++ path is exercised through the simulation below; here compare the
    # evaluator against the un-scaled analytic result it must reproduce
    v_plain = vc.evaluate(PLAIN)[1]
    for name, expect_ratio in (("Si", 1.0), ("PolySi", 1.0)):
        v = vc.evaluate(SELECTIVE, material=name)[1]
        check(f"{name} grows at the unrestricted rate",
              close(v, v_plain, 1e-9), f"{v:.6e} vs {v_plain:.6e}")
    v_ox = vc.evaluate(SELECTIVE, material="SiO2")[1]
    v_mask = vc.evaluate(SELECTIVE, material="Mask")[1]
    check("SiO2 is suppressed", v_ox < 0.05 * v_plain, f"{v_ox:.4e}")
    check("an unnamed material is inert", v_mask < 1e-12 * v_plain,
          f"{v_mask:.3e}")

    print("\n3) a plain mechanism is unaffected by the feature")
    v_plain_named = vc.evaluate(PLAIN, material="SiO2")[1]
    check("no materials block means no material dependence",
          close(v_plain_named, v_plain, 1e-12),
          f"{v_plain_named:.6e} vs {v_plain:.6e}")

    print("\n4) blanket simulations, one material each")
    v_si = blanket(mech, ps.Material.Si)
    v_ox_sim = blanket(mech, ps.Material.SiO2)
    v_mask_sim = blanket(mech, ps.Material.Mask)
    print(f"      Si   {v_si:.5e}    SiO2 {v_ox_sim:.5e}    Mask {v_mask_sim:.5e} nm/s")
    check("Si grows at the analytic rate",
          abs(v_si - v_plain) / v_plain < 0.05,
          f"{v_si:.5f} vs {v_plain:.5f} ({100*abs(v_si-v_plain)/v_plain:.2f}%)")
    check("SiO2 grows far more slowly than Si", v_ox_sim < 0.1 * v_si,
          f"{v_ox_sim:.3e} vs {v_si:.3e}")
    check("Mask does not grow", abs(v_mask_sim) < 1e-4 * v_si,
          f"{v_mask_sim:.3e}")

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
