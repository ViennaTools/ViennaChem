// Silane CVD of silicon: SiH4(g) -> Si(s) + 2 H2(g)
//
//   [R1] SiH4(g) + 2*  ->  SiH3* + H*          r1 = s(T) * Gamma * theta_*^2
//   [R2] SiH3*         ->  Si(s) + H* + H2(g)  r2 = k_inc * theta_SiH3
//   [R3] 2 H*          ->  H2(g) + 2*          r3 = k_d   * theta_H^2
//
// Set D = 2 for a trench, D = 3 for a cylindrical hole.

#include <geometries/psMakeHole.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psSilaneDeposition.hpp>

#include <process/psProcess.hpp>
#include <psUtil.hpp>

using namespace viennaps;

int main() {
  using NumericType = double;
  constexpr int D = 2; // 2 -> trench, 3 -> cylindrical hole

  Logger::setLogLevel(LogLevel::INFO);
  omp_set_num_threads(16);

  units::Length::setUnit("nm");
  units::Time::setUnit("s");

  const NumericType gridDelta = 2.0;
  const NumericType xExtent = 200.0;
  const NumericType yExtent = 200.0;
  const NumericType featureWidth = 80.0;
  const NumericType featureDepth = 120.0;
  const NumericType processTime = 300.0; // s (~22 nm of film)

  // This model has a custom surface model and no GPU counterpart.
  const auto fluxEngine = FluxEngineType::CPU_DISK;

  auto geometry = Domain<NumericType, D>::New(gridDelta, xExtent, yExtent);
  if constexpr (D == 2) {
    MakeTrench<NumericType, D>(geometry, featureWidth, featureDepth, 0.0, 0.0,
                               0.0, false, Material::Si, Material::Mask)
        .apply();
  } else {
    MakeHole<NumericType, D>(geometry, featureWidth / 2.0, featureDepth, 0.0,
                             0.0, 0.0, HoleShape::QUARTER, Material::Si,
                             Material::Mask)
        .apply();
  }
  // the deposited film is a new (poly-Si) level set on top
  geometry->duplicateTopLevelSet(Material::PolySi);

  SilaneDepositionParameters<NumericType> params;
  params.temperature = 900.; // K
  params.silaneFlux = 1.0e3; // 1e15 /cm² /s
  params.s0 = 1.0;
  params.Ea_ads = 0.6; // eV
  params.A_inc = 6.8e12;
  params.Ea_inc = 1.5; // eV
  params.A_d = 6.8e12;
  params.Ea_d = 2.0; // eV

  NumericType thetaSiH3, thetaH, r2;
  params.steadyState(params.silaneFlux, thetaSiH3, thetaH, r2);
  std::cout << "blanket growth rate = " << r2 / params.rho_Si << " nm/s\n";

  auto model = SmartPointer<SilaneDeposition<NumericType, D>>::New(params);

  CoverageParameters coverageParams;
  coverageParams.tolerance = 1e-4;
  RayTracingParameters rayTracingParams;
  rayTracingParams.raysPerPoint = 1000;

  geometry->saveSurfaceMesh("silane_initial.vtp");

  Process<NumericType, D> process(geometry, model);
  process.setProcessDuration(processTime);
  process.setParameters(coverageParams);
  process.setParameters(rayTracingParams);
  process.setFluxEngineType(fluxEngine);
  process.apply();

  geometry->saveSurfaceMesh("silane_final.vtp", true);
}
