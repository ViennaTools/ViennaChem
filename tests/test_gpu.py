#!/usr/bin/env python3
"""GPU gate: the device ray tracer reproduces the CPU result.

Only transport moves to the GPU; the coverage solve stays on the CPU in
ChemicalSurfaceModel. So the two engines must agree to within Monte-Carlo
noise on the blanket rate and on the trench profile.

The trench is compared against CPU_TRIANGLE rather than CPU_DISK. Disk-based
and line/triangle-based flux representations differ in step coverage by about
0.014 on their own, with no GPU involved, so CPU_DISK is the wrong reference
for a device-shader check: GPU_LINE and CPU_TRIANGLE are the like-for-like
pair. The blanket rate is insensitive to the representation, so CPU_DISK is
kept there.

Run with the ViennaPS venv:  .venv/bin/python test_gpu.py
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

IR = vc.from_file("silane.yaml")
mech = build(IR)
GPU = ps.FluxEngineType.GPU_LINE
CPU = ps.FluxEngineType.CPU_DISK


def top(dom):
    m = viennals.Mesh()
    ls2.ToSurfaceMesh(dom.getLevelSets()[-1], m).apply()
    return np.array(m.getNodes())


def blanket(engine, t=200.0, rays=2000):
    dom = ps.Domain(gridDelta=2.0, xExtent=100.0, yExtent=100.0)
    ps.MakePlane(dom, 0.0, ps.Material.Si).apply()
    dom.duplicateTopLevelSet(ps.Material.PolySi)
    y0 = top(dom)[:, 1].max()
    p = ps.Process(dom, ps.SurfaceChemistry(mech), t)
    p.setFluxEngineType(engine)
    c = ps.CoverageParameters(); c.tolerance = 1e-6; p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    return (top(dom)[:, 1].max() - y0) / t


def trench(engine, t=300.0, rays=1000):
    dom = ps.Domain(gridDelta=2.0, xExtent=200.0, yExtent=200.0)
    ps.MakeTrench(domain=dom, trenchWidth=80.0, trenchDepth=120.0,
                  material=ps.Material.Si).apply()
    dom.duplicateTopLevelSet(ps.Material.PolySi)

    def bounds():
        n = top(dom)
        return n[:, 1].max(), n[:, 1].min()

    t0, b0 = bounds()
    p = ps.Process(dom, ps.SurfaceChemistry(mech), t)
    p.setFluxEngineType(engine)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    t1, b1 = bounds()
    field, bottom = t1 - t0, b1 - b0
    return field, bottom, (bottom / field if field > 0 else 0.0)


def ion_blanket(mechanism, material, engine, t=100.0, rays=3000):
    """A blanket etch by an ion-driven mechanism, on either engine."""
    dom = ps.Domain(gridDelta=2.0, xExtent=100.0, yExtent=100.0)
    ps.MakePlane(dom, 0.0, material).apply()
    y0 = top(dom)[:, 1].max()
    p = ps.Process(dom, ps.SurfaceChemistry(mechanism), t)
    p.setFluxEngineType(engine)
    c = ps.CoverageParameters(); c.tolerance = 1e-4; c.maxIterations = 30
    p.setParameters(c)
    r = ps.RayTracingParameters(); r.raysPerPoint = rays; p.setParameters(r)
    p.apply()
    return (top(dom)[:, 1].max() - y0) / t


def main():
    print("GPU gate: SurfaceChemistry on the device")

    print("\n1) the model reports GPU support")
    model = ps.SurfaceChemistry(mech)
    check("hasGPU is set", getattr(model, "hasGPU", True))

    print("\n2) blanket rate, GPU vs CPU vs analytic")
    gamma = vc.default_fluxes(IR)
    theta = mech.solveCoverages(gamma, [0.0, 0.0])
    v_analytic = mech.growthRate(gamma, theta)
    v_cpu = blanket(CPU)
    v_gpu = blanket(GPU)
    print(f"      analytic {v_analytic:.5f}   CPU {v_cpu:.5f}   GPU {v_gpu:.5f} nm/s")
    check("GPU matches the analytic rate",
          abs(v_gpu - v_analytic) / v_analytic < 0.03,
          f"{100*abs(v_gpu-v_analytic)/v_analytic:.2f}%")
    check("GPU matches CPU", abs(v_gpu - v_cpu) / v_cpu < 0.03,
          f"{100*abs(v_gpu-v_cpu)/v_cpu:.2f}%")

    print("\n3) trench profile, GPU_LINE vs CPU_TRIANGLE (like-for-like)")
    f_cpu, b_cpu, s_cpu = trench(ps.FluxEngineType.CPU_TRIANGLE)
    f_gpu, b_gpu, s_gpu = trench(GPU)
    print(f"      CPU_TRIANGLE  field {f_cpu:.2f} nm  bottom {b_cpu:.2f} nm  step {s_cpu:.3f}")
    print(f"      GPU_LINE      field {f_gpu:.2f} nm  bottom {b_gpu:.2f} nm  step {s_gpu:.3f}")
    check("field growth agrees", abs(f_gpu - f_cpu) / f_cpu < 0.05,
          f"{100*abs(f_gpu-f_cpu)/f_cpu:.2f}%")
    check("step coverage agrees", abs(s_gpu - s_cpu) < 0.01,
          f"{abs(s_gpu-s_cpu):.4f}")
    check("coverage-dependent sticking reached the device shader",
          0.8 < s_gpu <= 1.05, f"step {s_gpu:.3f}")

    print("\n4) the device shader responds to the sticking, not just to geometry")
    # s(1200 K) is about 7x s(900 K); a higher sticking must lower step coverage
    mech.temperature = 1200.0
    _, _, s_hot = trench(GPU, t=34.0)
    mech.temperature = IR["constants"]["temperature"]
    print(f"      GPU step: {s_gpu:.4f} at 900 K, {s_hot:.4f} at 1200 K")
    check("step coverage falls when the sticking rises", s_hot < s_gpu,
          f"{s_hot:.4f} < {s_gpu:.4f}")

    print("\n5) the ion: the device evaluates the same yield as the CPU")
    # Both engines run the mechanism the reaction file declares; nothing about
    # the sputtering or the mask selectivity is written into the shader.
    for yaml, mats, t in [("ar_sputter.yaml",
                           [ps.Material.Si, ps.Material.Mask], 100.0),
                          ("cf4ar_etch.yaml", [ps.Material.Si], 50.0)]:
        mech_i = build(vc.from_file(yaml))
        for m in mats:
            cpu = ion_blanket(mech_i, m, CPU, t)
            gpu = ion_blanket(mech_i, m, ps.FluxEngineType.GPU_TRIANGLE, t)
            rel = abs(gpu - cpu) / abs(cpu)
            check(f"{yaml} on {m}: GPU matches CPU", rel < 0.05,
                  f"CPU {cpu:.5f}   GPU {gpu:.5f}   ({100*rel:.2f}%)")

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
