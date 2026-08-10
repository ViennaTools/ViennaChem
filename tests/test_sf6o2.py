#!/usr/bin/env python3
"""The acceptance test: regenerate a hand-written plasma model from a file.

`sf6o2.yaml` is ViennaPS's own SF6/O2 silicon etch (`psSF6O2Etching.hpp` +
`psPlasmaEtching.hpp`) written as reactions and rate constants. This compares
the generated mechanism against that hand-written C++ on the same geometry.

Unlike the silane check, whose reference was written alongside the generator,
this one is independent: `psSF6O2Etching` is ViennaPS's own model. It is also a
stronger check than an analytic one,
because it exercises the transport, the ion, the coverages and the velocity
together.

ONE CONVENTION DIFFERS. The reference records the raw incident flux and applies
the sticking beta only to the particle's re-emission, so its coverage balance
reads `Gamma_rec*(1-theta)`. The framework applies the sticking in BOTH places,
`s*Gamma_rec*(1-theta)`, which is the physically consistent choice: on a blanket
it makes the adsorption rate equal the source flux, whatever s, whereas the
reference gives source/beta. The two coincide exactly when beta = 1, so the
comparison below is run with beta = 1 on both sides. The default-beta case is
reported afterwards, with the 1/beta factor it predicts.

Run with the ViennaPS venv:  .venv/bin/python test_sf6o2.py
"""

import copy
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

IR = vc.from_file("sf6o2.yaml")
RAYS = 3000


def matched_ir():
    """The same mechanism with both stickings at 1, where the conventions agree."""
    out = copy.deepcopy(IR)
    for g in out["gas"]:
        if g["sticking"] is not None:
            g["sticking"]["s0"] = 1.0
    for rx in out["reactions"]:
        if rx["constant"]["kind"] == "sticking":
            rx["constant"]["s0"] = 1.0
    return out


def reference(beta=1.0):
    """ViennaPS's own SF6O2 model, at its defaults but for the sticking."""
    model = ps.SF6O2Etching()
    p = model.getParameters()
    p.beta_E.set(ps.Material.Si, beta)
    p.beta_E.set(ps.Material.Mask, beta)
    p.beta_P.set(ps.Material.Si, beta)
    p.beta_P.set(ps.Material.Mask, beta)
    model.setParameters(p)
    return model


def surface(dom):
    m = viennals.Mesh()
    ls2.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
    return np.array(m.getNodes())


def blanket(model, material, t=1.0, rays=RAYS):
    dom = ps.Domain(gridDelta=2.0, xExtent=100.0, yExtent=100.0)
    ps.MakePlane(dom, 0.0, material).apply()
    y0 = surface(dom)[:, 1].max()
    p = ps.Process(dom, model, t)
    p.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 30
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    return (surface(dom)[:, 1].max() - y0) / t


def trench(model, t=2.0, rays=1000):
    """A masked trench: depth, and the width at half depth."""
    dom = ps.Domain(gridDelta=2.0, xExtent=200.0, yExtent=200.0)
    ps.MakeTrench(domain=dom, trenchWidth=60.0, trenchDepth=40.0,
                  maskHeight=30.0, material=ps.Material.Si,
                  maskMaterial=ps.Material.Mask).apply()
    n0 = surface(dom)
    p = ps.Process(dom, model, t)
    p.setFluxEngineType(ps.FluxEngineType.CPU_DISK)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 30
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    n1 = surface(dom)
    depth = n0[:, 1].min() - n1[:, 1].min()      # how much deeper the floor is
    top = n1[:, 1].max() - n0[:, 1].max()        # mask erosion (negative)
    # sidewall position at mid depth, on the right-hand wall
    mid = 0.5 * (n1[:, 1].min() + n0[:, 1].min())
    band = n1[np.abs(n1[:, 1] - mid) < 4.0]
    half = band[band[:, 0] > 0][:, 0].min() if len(band[band[:, 0] > 0]) else np.nan
    return depth, top, half


def main():
    print("Acceptance test: SF6/O2 from a reaction file vs the hand-written model")

    print("\n1) the reaction file carries the published, non-mass-action orders")
    r3 = IR["reactions"][2]
    check("the chemical etch consumes four F but is first order",
          r3["covFactors"] == [{"idx": 0, "exp": 1}] and r3["nu"][0] == -4.0,
          f"exp {r3['covFactors'][0]['exp']}, nu {r3['nu'][0]}")
    r4 = IR["reactions"][3]
    check("the ion-enhanced step consumes two F but is first order",
          r4["covFactors"] == [{"idx": 0, "exp": 1}] and r4["nu"][0] == -2.0,
          f"exp {r4['covFactors'][0]['exp']}, nu {r4['nu'][0]}")
    check("the mask has its own solid density",
          IR["solid"]["rhoMaterials"] == {"Mask": 500.0},
          str(IR["solid"]["rhoMaterials"]))

    print("\n2) front end, data evaluator and C++ agree on the mechanism")
    mech = build(IR)
    gamma = vc.mechanism_fluxes(IR)
    th_py = vc.solve_coverages(IR)
    th_cpp = mech.solveCoverages(gamma, [0.0] * len(IR["coverages"]))
    check("coverages agree",
          max(abs(a - b) for a, b in zip(th_py, th_cpp)) < 1e-12,
          f"theta_F {th_cpp[0]:.6f}, theta_O {th_cpp[1]:.6f}")
    v_py = vc.growth_rate(IR, th_py)
    v_cpp = mech.growthRate(gamma, th_cpp)
    check("rates agree", abs(v_py - v_cpp) < 1e-10, f"{v_cpp:.6f} nm/s")

    print("\n3) blanket etch rate vs the hand-written model (matched convention)")
    mech_m = build(matched_ir())
    for material, tol in ((ps.Material.Si, 0.03), (ps.Material.Mask, 0.05)):
        ref = blanket(reference(1.0), material)
        gen = blanket(ps.SurfaceChemistry(mech_m), material)
        rel = abs(gen - ref) / abs(ref)
        check(f"{material}: generated matches hand-written", rel < tol,
              f"reference {ref:.5f}   generated {gen:.5f}   ({100*rel:.2f}%)")

    print("\n4) masked trench profile vs the hand-written model")
    d_ref, t_ref, h_ref = trench(reference(1.0))
    d_gen, t_gen, h_gen = trench(ps.SurfaceChemistry(mech_m))
    print(f"      reference  depth {d_ref:7.3f}  mask {t_ref:7.3f}  wall x {h_ref:7.3f}")
    print(f"      generated  depth {d_gen:7.3f}  mask {t_gen:7.3f}  wall x {h_gen:7.3f}")
    check("trench depth agrees", abs(d_gen - d_ref) / abs(d_ref) < 0.05,
          f"{100*abs(d_gen-d_ref)/abs(d_ref):.2f}%")
    check("mask erosion agrees", abs(t_gen - t_ref) < 0.5 + 0.1 * abs(t_ref),
          f"{t_ref:.3f} vs {t_gen:.3f}")
    check("sidewall position agrees", abs(h_gen - h_ref) < 2.0,
          f"{h_ref:.3f} vs {h_gen:.3f} nm")

    print("\n5) the convention difference, at the model's own beta = 0.7")
    ref07 = blanket(reference(0.7), ps.Material.Si)
    gen07 = blanket(ps.SurfaceChemistry(build(IR)), ps.Material.Si)
    print(f"      reference {ref07:.5f}   generated {gen07:.5f} nm/s"
          f"   ratio {ref07/gen07:.3f}")
    check("the reference etches faster, as 1/beta on the etchant supply predicts",
          1.0 < ref07 / gen07 < 1.0 / 0.7 + 0.1,
          f"{ref07/gen07:.3f}, with 1/beta = {1/0.7:.3f} the upper bound")

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
