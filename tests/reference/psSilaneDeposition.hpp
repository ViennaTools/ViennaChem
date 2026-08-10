#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include "../process/psProcessModel.hpp"
#include "../process/psSurfaceModel.hpp"
#include "../process/psVelocityField.hpp"
#include "../psConstants.hpp"
#include "../psUnits.hpp"

#include <unordered_map>

namespace viennaps {

using namespace viennacore;

// Silane CVD of silicon (poly-Si LPCVD), generated from the reaction mechanism
//
//   Overall:  SiH4(g) -> Si(s) + 2 H2(g)
//
//   [R1] SiH4(g) + 2*  ->  SiH3* + H*          r1 = s(T) * Gamma * theta_*^2
//   [R2] SiH3*         ->  Si(s) + H* + H2(g)  r2 = k_inc * theta_SiH3
//   [R3] 2 H*          ->  H2(g) + 2*          r3 = k_d   * theta_H^2
//
// Every step balances in atoms and in surface sites (the total site density is
// constant; the deposited Si joins the lattice), and [R1]+[R2]+[R3] sums to the
// overall reaction.
//
// Nothing here is a fitted "sticking coefficient": the adsorption rate constant
// of the elementary step [R1] is expressed, as usual for a gas-surface
// collision process, as a sticking probability
//
//   s(T) = s0 * exp(-Ea_ads / kB T)          (activated dissociative chemisorption)
//
// which is equivalent to an adsorption rate constant via s = 4 k_ads / vbar
// (see stickingFromRateConstant below).
//
// Steady-state coverage solve. With theta_* = 1 - theta_SiH3 - theta_H, the
// balances are  r1 = r2  (SiH3*)  and  r1 + r2 = 2 r3  (H*), hence r1 = r2 = r3.
// Substituting theta_SiH3 = r1/k_inc and theta_H = sqrt(r1/k_d) into
// r1 = s*Gamma*theta_*^2 and setting u = sqrt(r1) gives a quadratic in u:
//
//   (C/k_inc) u^2 + (1 + C/sqrt(k_d)) u - C = 0,      C = sqrt(s*Gamma)
//
// solved stably as  u = 2C / (B + sqrt(B^2 + 4AC)).  The growth rate is then
//
//   v_n = (1/rho_Si) * r2                     [nm/s, positive = deposition]
//
// As k_inc -> infinity this reduces to the lumped two-reaction model with
//   theta_H = lambda/(1+lambda),  lambda = sqrt(s*Gamma/k_d).

template <typename NumericType> struct SilaneDepositionParameters {
  // process conditions
  NumericType temperature = 900.;  // K
  NumericType silaneFlux = 1.0e3;  // incident SiH4 flux, in 1e15 /cm² /s

  // [R1] dissociative chemisorption:  s(T) = s0 * exp(-Ea_ads / kB T)
  NumericType s0 = 1.0;     // sticking prefactor (1 = barrierless)
  NumericType Ea_ads = 0.6; // eV

  // The surface rate prefactors are areal: A ~ S0 * nu, with the site density
  // S0 ~ 0.68 (in 1e15 /cm²) and an attempt frequency nu ~ 1e13 /s, so
  // A ~ 6.8e12 in units of 1e15 /cm² /s.

  // [R2] incorporation:  k_inc = A_inc * exp(-Ea_inc / kB T), in 1e15 /cm² /s
  NumericType A_inc = 6.8e12;
  NumericType Ea_inc = 1.5; // eV

  // [R3] recombinative H2 desorption (2nd order):
  //      k_d = A_d * exp(-Ea_d / kB T), in 1e15 /cm² /s
  NumericType A_d = 6.8e12;
  NumericType Ea_d = 2.0; // eV  (H2 desorption barrier, ~190 kJ/mol)

  // silicon
  NumericType rho_Si = 5.02; // 1e22 atoms/cm³

  // optional operator-split surface diffusion of the coverages (length²/time)
  NumericType thetaHDiffusionCoefficient = 0.;
  NumericType thetaSiH3DiffusionCoefficient = 0.;

  // --- derived rate constants -------------------------------------------
  NumericType sticking() const {
    return s0 * std::exp(-Ea_ads / (constants::kB * temperature));
  }
  NumericType k_inc() const {
    return A_inc * std::exp(-Ea_inc / (constants::kB * temperature));
  }
  NumericType k_des() const {
    return A_d * std::exp(-Ea_d / (constants::kB * temperature));
  }

  // Convert an adsorption rate constant into a sticking probability:
  //   s = 4 k_ads / vbar,   vbar = sqrt(8 kB T / pi m)
  // molarMass in amu, k_ads in cm/s (i.e. k_ads * n_gas = flux).
  static NumericType stickingFromRateConstant(NumericType k_ads,
                                              NumericType temperatureK,
                                              NumericType molarMass) {
    // gasMeanThermalVelocity takes Celsius and returns um/s
    const double vbar_um_s = constants::gasMeanThermalVelocity(
        temperatureK - 273.15, static_cast<double>(molarMass));
    const double vbar_cm_s = vbar_um_s * 1e-4;
    return static_cast<NumericType>(4. * k_ads / vbar_cm_s);
  }

  // Steady-state solution of the coverage balance at one surface point.
  // gamma: local incident SiH4 flux in 1e15 /cm² /s.
  void steadyState(NumericType gamma, NumericType &thetaSiH3,
                   NumericType &thetaH, NumericType &r2) const {
    const NumericType kInc = k_inc();
    const NumericType kD = k_des();
    const NumericType sG = sticking() * gamma;

    if (sG <= 0. || kInc <= 0. || kD <= 0.) {
      thetaSiH3 = 0.;
      thetaH = (kD <= 0. && sG > 0.) ? 1. : 0.;
      r2 = 0.;
      return;
    }

    const NumericType C = std::sqrt(sG);
    const NumericType B = 1. + C / std::sqrt(kD);
    const NumericType A = C / kInc;
    // stable root of A u² + B u - C = 0
    const NumericType u = 2. * C / (B + std::sqrt(B * B + 4. * A * C));

    const NumericType r1 = u * u;
    thetaSiH3 = std::min(std::max(r1 / kInc, NumericType(0.)), NumericType(1.));
    thetaH = std::min(std::max(u / std::sqrt(kD), NumericType(0.)),
                      NumericType(1.));
    r2 = kInc * thetaSiH3; // = r1 at steady state; the Si-forming rate
  }

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> metaData;
    metaData["Temperature"] = {temperature};
    metaData["SilaneFlux"] = {silaneFlux};
    metaData["s0"] = {s0};
    metaData["Ea_ads"] = {Ea_ads};
    metaData["A_inc"] = {A_inc};
    metaData["Ea_inc"] = {Ea_inc};
    metaData["A_d"] = {A_d};
    metaData["Ea_d"] = {Ea_d};
    metaData["RhoSi"] = {rho_Si};
    metaData["Sticking"] = {sticking()};
    metaData["k_inc"] = {k_inc()};
    metaData["k_des"] = {k_des()};
    if (thetaHDiffusionCoefficient > 0.)
      metaData["thetaHDiffusionCoefficient"] = {thetaHDiffusionCoefficient};
    if (thetaSiH3DiffusionCoefficient > 0.)
      metaData["thetaSiH3DiffusionCoefficient"] = {
          thetaSiH3DiffusionCoefficient};
    return metaData;
  }
};

namespace impl {

template <typename NumericType, int D>
class SilaneSurfaceModel : public SurfaceModel<NumericType> {
public:
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;
  const SilaneDepositionParameters<NumericType> &params;

  explicit SilaneSurfaceModel(
      const SilaneDepositionParameters<NumericType> &pParams)
      : params(pParams) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = PointData<NumericType>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints, 0.);
    coverages->insertNextScalarData(cov, "thetaSiH3");
    coverages->insertNextScalarData(cov, "thetaH");
  }

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    if (Logger::hasIntermediate()) {
      if (surfaceData == nullptr) {
        surfaceData = PointData<NumericType>::New();
      } else {
        surfaceData->clear();
      }
      std::vector<NumericType> data(numGeometryPoints, 0.);
      surfaceData->insertNextScalarData(data, "growthRate");
    }
  }

  void updateCoverages(SmartPointer<PointData<NumericType>> fluxes,
                       const std::vector<NumericType> &materialIds) override {
    const auto numPoints = materialIds.size();
    const auto silaneFlux = fluxes->getScalarData("silaneFlux");

    auto thetaSiH3 = coverages->getScalarData("thetaSiH3");
    auto thetaH = coverages->getScalarData("thetaH");
    thetaSiH3->resize(numPoints);
    thetaH->resize(numPoints);

#pragma omp parallel for
    for (size_t i = 0; i < numPoints; ++i) {
      const auto gamma = silaneFlux->at(i) * params.silaneFlux;
      NumericType tS, tH, r2;
      params.steadyState(gamma, tS, tH, r2);
      thetaSiH3->at(i) = tS;
      thetaH->at(i) = tH;
    }
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    const auto numPoints = materialIds.size();
    std::vector<NumericType> velocity(numPoints, 0.);

    const auto thetaSiH3 = coverages->getScalarData("thetaSiH3");

    std::vector<NumericType> *grRate = nullptr;
    if (Logger::hasIntermediate()) {
      grRate = surfaceData->getScalarData("growthRate");
      grRate->resize(numPoints);
    }

    // r in 1e15 /cm²/s divided by rho in 1e22 /cm³ gives nm/s directly
    const double unitConversion =
        units::Time::convertSecond() / units::Length::convertNanometer();
    const auto kInc = params.k_inc();

    #pragma omp parallel for
    for (size_t i = 0; i < numPoints; ++i) {
      // solid-forming reaction [R2]: one Si per event
      const auto r2 = kInc * thetaSiH3->at(i);
      velocity[i] = (1. / params.rho_Si) * r2 * unitConversion; // deposition > 0
      if (Logger::hasIntermediate())
        grRate->at(i) = velocity[i];
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(velocity));
  }

  std::optional<std::unordered_map<std::string, NumericType>>
  getDiffusionCoefficients() const override {
    std::unordered_map<std::string, NumericType> coefficients;
    if (params.thetaHDiffusionCoefficient > 0.)
      coefficients["thetaH"] = params.thetaHDiffusionCoefficient;
    if (params.thetaSiH3DiffusionCoefficient > 0.)
      coefficients["thetaSiH3"] = params.thetaSiH3DiffusionCoefficient;
    if (coefficients.empty())
      return std::nullopt;
    return coefficients;
  }
};

// Neutral SiH4. Its sticking is the rate of the elementary dissociative
// chemisorption [R1]: two free sites are required, so S_eff = s(T)*theta_*².
template <typename NumericType, int D>
class SilaneParticle final
    : public viennaray::Particle<SilaneParticle<NumericType, D>, NumericType> {
  const SilaneDepositionParameters<NumericType> &params;

public:
  explicit SilaneParticle(
      const SilaneDepositionParameters<NumericType> &pParams)
      : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int, PointData<NumericType> &localData,
                        const PointData<NumericType> *, RNG &) override {
    // record the raw incident flux; the sticking enters the balance analytically
    localData.addToScalarData(0, primID, rayWeight);
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int,
                    const PointData<NumericType> *globalData,
                    RNG &rngState) override {
    const auto &thetaSiH3 = globalData->getScalarData(0)->at(primID);
    const auto &thetaH = globalData->getScalarData(1)->at(primID);
    const NumericType thetaFree =
        std::max(NumericType(1.) - thetaSiH3 - thetaH, NumericType(0.));
    const NumericType S_eff = params.sticking() * thetaFree * thetaFree;

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }

  NumericType getSourceDistributionPower() const override { return 1.; }
  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {"silaneFlux"};
  }
};
} // namespace impl

/// Silane CVD of silicon: SiH4(g) -> Si(s) + 2 H2(g).
template <typename NumericType, int D>
class SilaneDeposition final : public ProcessModelCPU<NumericType, D> {
public:
  SilaneDeposition() { initializeModel(); }

  explicit SilaneDeposition(
      const SilaneDepositionParameters<NumericType> &pParams)
      : params(pParams) {
    initializeModel();
  }

  void setParameters(const SilaneDepositionParameters<NumericType> &pParams) {
    params = pParams;
    initializeModel();
  }

  SilaneDepositionParameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    if (units::Length::getUnit() == units::Length::UNDEFINED ||
        units::Time::getUnit() == units::Time::UNDEFINED) {
      VIENNACORE_LOG_ERROR("Units have not been set.");
    }

    auto silane = std::make_unique<impl::SilaneParticle<NumericType, D>>(params);

    auto surfModel =
        SmartPointer<impl::SilaneSurfaceModel<NumericType, D>>::New(params);
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SilaneDeposition");
    this->particles.clear();
    this->insertNextParticleType(silane);

    this->processMetaData = params.toProcessMetaData();
    this->processMetaData["Units"] = std::vector<double>{
        static_cast<double>(units::Length::getInstance().getUnit()),
        static_cast<double>(units::Time::getInstance().getUnit())};
  }

  SilaneDepositionParameters<NumericType> params;
};

PS_PRECOMPILE_PRECISION_DIMENSION(SilaneDeposition)

} // namespace viennaps
