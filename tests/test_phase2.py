#!/usr/bin/env python3
"""Phase 2 gate: the generic ViennaPS SurfaceChemistry reproduces the silane model.

Checks, in order of increasing integration:

  1. the C++ mechanism evaluates coverages and growth rate identically to the
     pure-Python IR evaluator (same algorithm, two implementations),
  2. it matches a by-hand implementation of the same mechanism (the values
     frozen below, from tests/reference/psSilaneDeposition.hpp),
  3. a blanket (flat wafer) simulation reproduces the analytic growth rate,
  4. the coverage-dependent sticking reaches the particle, so a trench shows the
     expected step coverage.

Run with the ViennaPS venv:  .venv/bin/python test_phase2.py
"""

import sys

import viennaps as ps

import viennachem as vc
from support import build

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def close(a, b, rtol=1e-9):
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


REF_THETA_SIH3 = 1.329904e-05
REF_THETA_H = 9.159010e-02
REF_V = 0.071779

ps.setDimension(2)
ps.setNumThreads(16)
ps.Length.setUnit("nm")
ps.Time.setUnit("s")
ps.Logger.setLogLevel(ps.LogLevel.WARNING)

IR = vc.from_file("silane.yaml")


def test_mechanism_matches_python():
    print("\n1) C++ mechanism vs the pure-Python IR evaluator")
    mech = build(IR)

    gamma = vc.default_fluxes(IR)
    ncov = len(IR["coverages"])

    theta_cpp = mech.solveCoverages(gamma, [0.0] * ncov)
    v_cpp = mech.growthRate(gamma, theta_cpp)
    theta_py, v_py = vc.evaluate(IR)

    for c, a, b in zip(IR["coverages"], theta_cpp, theta_py):
        check(f"theta_{c['name']}", close(a, b, 1e-7), f"C++ {a:.6e} vs py {b:.6e}")
    check("v_n", close(v_cpp, v_py, 1e-7), f"C++ {v_cpp:.6f} vs py {v_py:.6f}")

    ks_cpp = mech.rateConstants()
    ks_py = vc.rate_constants(IR)
    check("rate constants", all(close(a, b, 1e-10) for a, b in zip(ks_cpp, ks_py)))
    check("sticking of SiH4", close(mech.stickingOf(0), vc.sticking(IR, "SiH4")[0], 1e-10),
          f"{mech.stickingOf(0):.6e}")
    return mech


def test_against_reference(mech):
    print("\n2) C++ mechanism vs the by-hand implementation")
    gamma = vc.default_fluxes(IR)
    theta = mech.solveCoverages(gamma, [0.0, 0.0])
    v = mech.growthRate(gamma, theta)
    check("theta_SiH3", close(theta[0], REF_THETA_SIH3, 1e-5), f"{theta[0]:.6e}")
    check("theta_H", close(theta[1], REF_THETA_H, 1e-6), f"{theta[1]:.6e}")
    check("v_n [nm/s]", close(v, REF_V, 1e-5), f"{v:.6f}")


def blanket_rate(model, t=200.0, rays=2000):
    """Deposit on a flat wafer and measure the film thickness per unit time."""
    import numpy as np
    import viennals
    import viennals.d2 as ls

    dom = ps.Domain(gridDelta=2.0, xExtent=100.0, yExtent=100.0)
    ps.MakePlane(dom, 0.0, ps.Material.Si).apply()
    dom.duplicateTopLevelSet(ps.Material.PolySi)

    def top():
        m = viennals.Mesh()
        ls.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
        return np.array(m.getNodes())[:, 1].max()

    y0 = top()
    proc = ps.Process(dom, model, t)
    proc.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    cov = ps.CoverageParameters(); cov.tolerance = 1e-6; proc.setParameters(cov)
    ray = ps.RayTracingParameters(); ray.raysPerPoint = rays; proc.setParameters(ray)
    proc.apply()
    return (top() - y0) / t


def test_blanket(mech):
    print("\n3) blanket simulation vs the analytic growth rate (the Phase 2 gate)")
    gamma = vc.default_fluxes(IR)
    theta = mech.solveCoverages(gamma, [0.0, 0.0])
    v_analytic = mech.growthRate(gamma, theta)

    model = ps.SurfaceChemistry(mech)
    v_sim = blanket_rate(model)
    err = 100.0 * abs(v_sim - v_analytic) / v_analytic
    check("blanket rate within Monte-Carlo noise", err < 2.0,
          f"sim {v_sim:.5f} vs analytic {v_analytic:.5f} nm/s  ({err:.2f}%)")


def test_temperature_sweep(mech):
    print("\n4) both limiting regimes are reproduced")
    gamma = vc.default_fluxes(IR)
    rows = []
    for T in (600.0, 900.0, 1200.0):
        mech.temperature = T
        theta = mech.solveCoverages(gamma, [0.0, 0.0])
        v = mech.growthRate(gamma, theta)
        rows.append((T, theta[1], v))
        print(f"      T={T:6.0f} K   theta_H={theta[1]:.4f}   v={v:.4e} nm/s")
    mech.temperature = IR["constants"]["temperature"]
    check("theta_H falls as T rises", rows[0][1] > rows[1][1] > rows[2][1])
    check("growth rate rises with T", rows[0][2] < rows[1][2] < rows[2][2])


def test_trench(mech):
    print("\n5) trench: coverage-dependent sticking reaches the particle")
    import numpy as np
    import viennals
    import viennals.d2 as ls

    dom = ps.Domain(gridDelta=2.0, xExtent=200.0, yExtent=200.0)
    ps.MakeTrench(domain=dom, trenchWidth=80.0, trenchDepth=120.0,
                  material=ps.Material.Si).apply()
    dom.duplicateTopLevelSet(ps.Material.PolySi)

    def bounds():
        m = viennals.Mesh()
        ls.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
        n = np.array(m.getNodes())
        return n[:, 1].max(), n[:, 1].min()

    t0, b0 = bounds()
    proc = ps.Process(dom, ps.SurfaceChemistry(mech), 300.0)
    proc.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    cov = ps.CoverageParameters(); cov.tolerance = 1e-4; proc.setParameters(cov)
    ray = ps.RayTracingParameters(); ray.raysPerPoint = 1000; proc.setParameters(ray)
    proc.apply()
    t1, b1 = bounds()

    field, bottom = t1 - t0, b1 - b0
    step = bottom / field if field > 0 else 0.0
    print(f"      field {field:.2f} nm, bottom {bottom:.2f} nm, step coverage {step:.3f}")
    check("film grew on the field", field > 0.0)
    check("film grew at the trench bottom", bottom > 0.0)
    check("step coverage is high (low s_eff)", 0.8 < step <= 1.05, f"{step:.3f}")


def main():
    print("Phase 2 gate: generic SurfaceChemistry")
    mech = test_mechanism_matches_python()
    test_against_reference(mech)
    test_blanket(mech)
    test_temperature_sweep(mech)
    test_trench(mech)

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
