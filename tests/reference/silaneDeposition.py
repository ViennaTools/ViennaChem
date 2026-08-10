# Silane CVD of silicon: SiH4(g) -> Si(s) + 2 H2(g)
#
# Generated directly from the elementary surface mechanism
#
#   [R1] SiH4(g) + 2*  ->  SiH3* + H*          r1 = s(T) * Gamma * theta_*^2
#   [R2] SiH3*         ->  Si(s) + H* + H2(g)  r2 = k_inc * theta_SiH3
#   [R3] 2 H*          ->  H2(g) + 2*          r3 = k_d   * theta_H^2
#
# Two coverages (theta_SiH3, theta_H); the adsorption "sticking" is the
# elementary rate constant s(T) = s0 * exp(-Ea_ads / kB T), not a fitted number.
#
# Because s_eff = s(T) * theta_*^2 is coverage dependent, sticking varies over
# the feature and the growth is a coupled transport/chemistry problem.
#
# Geometry:  2D (default) -> trench,  3D -> cylindrical hole
#   python silaneDeposition.py -D 3

from argparse import ArgumentParser

import viennaps as ps

parser = ArgumentParser(
    prog="silaneDeposition", description="Silane CVD of Si (2D trench or 3D hole)."
)
parser.add_argument("-D", "--dim", dest="dim", type=int, default=2, choices=(2, 3))
args = parser.parse_args()

ps.setDimension(args.dim)
ps.setNumThreads(16)
ps.Logger.setLogLevel(ps.LogLevel.INFO)

ps.Length.setUnit("nm")
ps.Time.setUnit("s")

# --- geometry (nm) ---------------------------------------------------------
grid_delta = 2.0
x_extent = 200.0
y_extent = 200.0
feature_width = 80.0
feature_depth = 120.0

process_time = 300.0  # s  (~22 nm of film; keeps the trench open)

# This model has a custom surface model and no GPU counterpart, so it runs on
# the CPU flux engine.
flux_engine = ps.FluxEngineType.CPU_DISK


def make_geometry():
    domain = ps.Domain(gridDelta=grid_delta, xExtent=x_extent, yExtent=y_extent)
    if args.dim == 2:
        ps.MakeTrench(
            domain=domain,
            trenchWidth=feature_width,
            trenchDepth=feature_depth,
            material=ps.Material.Si,
        ).apply()
    else:
        ps.MakeHole(
            domain=domain,
            holeRadius=feature_width / 2.0,
            holeDepth=feature_depth,
            holeShape=ps.HoleShape.QUARTER,
            material=ps.Material.Si,
        ).apply()
    # the deposited film is a new (poly-Si) level set on top
    domain.duplicateTopLevelSet(ps.Material.PolySi)
    return domain


def parameters(temperature=900.0):
    p = ps.SilaneDepositionParameters()
    p.temperature = temperature  # K
    p.silaneFlux = 1.0e3  # 1e15 /cm^2 /s
    # [R1] dissociative chemisorption
    p.s0 = 1.0
    p.Ea_ads = 0.6  # eV
    # [R2] incorporation
    p.A_inc = 6.8e12
    p.Ea_inc = 1.5  # eV
    # [R3] recombinative H2 desorption (2nd order)
    p.A_d = 6.8e12
    p.Ea_d = 2.0  # eV
    return p


def main():
    p = parameters()
    thetaSiH3, thetaH, r2 = p.steadyState(p.silaneFlux)
    print(f"T = {p.temperature:.0f} K")
    print(f"  s(T)    = {p.sticking():.3e}")
    print(f"  k_inc   = {p.k_inc():.3e}   k_des = {p.k_des():.3e}  [1e15/cm2/s]")
    print(f"  theta_SiH3 = {thetaSiH3:.3e}   theta_H = {thetaH:.4f}")
    print(f"  blanket growth rate = {r2 / p.rho_Si:.4f} nm/s")

    domain = make_geometry()
    domain.saveSurfaceMesh(filename=f"silane_{args.dim}D_initial.vtp")

    process = ps.Process(domain, ps.SilaneDeposition(p), process_time)
    process.setFluxEngineType(flux_engine)

    cov = ps.CoverageParameters()
    cov.tolerance = 1e-4
    process.setParameters(cov)

    ray = ps.RayTracingParameters()
    ray.raysPerPoint = 1000
    process.setParameters(ray)

    process.apply()
    domain.saveSurfaceMesh(
        filename=f"silane_{args.dim}D_final.vtp", addInterfaces=True
    )


if __name__ == "__main__":
    main()
