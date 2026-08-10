#!/usr/bin/env python3
"""Multiple site types on the C++ model (and GPU).

The pure-Python evaluator was already checked against the SymPy front end for
the two-site GaAs toy (test_ir.py). Here the compiled ChemicalMechanism must
reproduce that evaluator, and the GPU flux engine must reproduce the CPU one, so
the per-type closure is correct through every layer.

Run with the ViennaPS venv:  .venv/bin/python test_multisite.py
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

IR = vc.from_file("gaas_toy.yaml")
mech = build(IR)
FILM = getattr(ps.Material, "GaAs", ps.Material.Metal)


def top(dom):
    m = viennals.Mesh()
    ls2.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
    return np.array(m.getNodes())


def blanket(engine, t=50.0, rays=3000):
    dom = ps.Domain(gridDelta=2.0, xExtent=100.0, yExtent=100.0)
    ps.MakePlane(dom, 0.0, ps.Material.Si).apply()
    dom.duplicateTopLevelSet(FILM)
    y0 = top(dom)[:, 1].max()
    p = ps.Process(dom, ps.SurfaceChemistry(mech), t)
    p.setFluxEngineType(engine)
    # Bound the outer flux<->coverage loop: its delta metric floors on Monte-Carlo
    # noise, so tolerance alone can leave it spinning (the default cap is 2^32-1).
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 20
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    return (top(dom)[:, 1].max() - y0) / t


def main():
    print("Multiple site types: C++ model and GPU vs the Python evaluator")

    print("\n1) the C++ mechanism reproduces the two-site Python evaluator")
    gamma = vc.default_fluxes(IR)
    ncov = len(IR["coverages"])
    theta_py, v_py = vc.evaluate(IR)

    theta_cpp = mech.solveCoverages(gamma, [0.0] * ncov)
    v_cpp = mech.growthRate(gamma, theta_cpp)
    for c, a, b in zip(IR["coverages"], theta_cpp, theta_py):
        check(f"theta_{c['name']}", close(a, b, 1e-6), f"C++ {a:.6e} vs py {b:.6e}")
    check("growth rate", close(v_cpp, v_py, 1e-6), f"C++ {v_cpp:.6f} vs py {v_py:.6f}")

    print("\n2) the coupled site balances hold: r1 = r2 = r3")
    ks = mech.rateConstants()
    frees = vc.free_fractions(IR, theta_cpp)
    rr = [vc.rate(rx, ks[j], gamma, theta_cpp, frees)
          for j, rx in enumerate(IR["reactions"])]
    check("r1 = r2 = r3", close(rr[0], rr[1], 1e-6) and close(rr[1], rr[2], 1e-6),
          " ".join(f"{r:.4e}" for r in rr))

    print("\n3) blanket rate: analytic vs CPU vs GPU")
    v_analytic = v_cpp
    v_cpu = blanket(ps.FluxEngineType.CPU_DISK)
    check("CPU blanket matches the analytic rate",
          abs(v_cpu - v_analytic) / v_analytic < 0.03,
          f"CPU {v_cpu:.5f} vs analytic {v_analytic:.5f} ({100*abs(v_cpu-v_analytic)/v_analytic:.2f}%)")
    try:
        v_gpu = blanket(ps.FluxEngineType.GPU_LINE)
        check("GPU blanket matches CPU (per-type shader)",
              abs(v_gpu - v_cpu) / v_cpu < 0.05,
              f"GPU {v_gpu:.5f} vs CPU {v_cpu:.5f} ({100*abs(v_gpu-v_cpu)/v_cpu:.2f}%)")
    except RuntimeError as e:
        check("GPU available", False, str(e)[:80])

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
