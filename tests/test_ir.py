#!/usr/bin/env python3
"""Phase 1 gate: the IR round-trips silane, and the derived quantities are correct.

Run:  python3 test_ir.py
"""

import json
import math
import os
import sys
import tempfile

import viennachem as vc
from support import build

FAIL = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def close(a, b, rtol=1e-9, atol=0.0):
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


SILANE = """
name: silane_polySi_LPCVD
temperature: 900.0
species:
  SiH4:  {phase: gas,     flux: 1.0e3}
  H2:    {phase: gas,     reemit_only: true}
  SiH3*: {phase: surface}
  H*:    {phase: surface}
  Si:    {phase: solid,   rho: 5.02}
surface:
  - {eq: "SiH4 -> SiH3* + H*",    type: adsorption, s0: 1.0,    Ea: 0.6}
  - {eq: "SiH3* -> Si + H* + H2", type: surface,    k0: 6.8e12, Ea: 1.5}
  - {eq: "2H* -> H2",             type: desorption, k0: 6.8e12, Ea: 2.0}
"""

# Eley-Rideal probe: gas atomic H abstracts adsorbed H. Exercises the Gamma*theta
# rate form and a non-1:1:1 null space.
ER = """
name: eley_rideal_probe
temperature: 900.0
species:
  SiH4:  {phase: gas, flux: 1.0e3}
  H:     {phase: gas, flux: 5.0e1}
  H2:    {phase: gas, reemit_only: true}
  SiH3*: {phase: surface}
  H*:    {phase: surface}
  Si:    {phase: solid, rho: 5.02}
surface:
  - {eq: "SiH4 -> SiH3* + H*",    type: adsorption, s0: 1.0,    Ea: 0.6}
  - {eq: "H + H* -> H2",          type: surface,    k0: 1.0e12, Ea: 0.1}
  - {eq: "SiH3* -> Si + H* + H2", type: surface,    k0: 6.8e12, Ea: 1.5}
"""

# Two site types: a toy GaAs surface where Ga adsorbs on cation sites (A) and As
# on anion sites (B), and a cross reaction deposits GaAs freeing one of each. It
# exercises one closure per type; the free fraction of A must not respond to a
# coverage on B.
TWOSITE = """
name: gaas_toy_twosite
temperature: 900.0
species:
  Ga:   {phase: gas, flux: 1.0e2}
  As:   {phase: gas, flux: 1.0e2}
  Ga*:  {phase: surface, site: A}
  As*:  {phase: surface, site: B}
  GaAs: {phase: solid, rho: 2.2}
surface:
  - {eq: "Ga -> Ga*",         type: adsorption, s0: 1.0,    Ea: 0.3}
  - {eq: "As -> As*",         type: adsorption, s0: 1.0,    Ea: 0.3}
  - {eq: "Ga* + As* -> GaAs", type: surface,    k0: 1.0e12, Ea: 1.0}
"""

# Reference values from a by-hand implementation of the same mechanism,
# kept in tests/reference/psSilaneDeposition.hpp
REF_THETA_SIH3 = 1.329904e-05
REF_THETA_H = 9.159010e-02
REF_V = 0.071779


def write_tmp(text):
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fh.write(text)
    fh.close()
    return fh.name


def test_silane_ir_structure():
    print("\n1) IR structure (silane)")
    path = write_tmp(SILANE)
    ir = vc.from_file(path)
    os.unlink(path)

    names = [c["name"] for c in ir["coverages"]]
    check("two coverages, in order", names == ["SiH3*", "H*"], str(names))

    gas = {g["name"]: g for g in ir["gas"]}
    check("SiH4 is traced", gas["SiH4"]["traced"])
    check("H2 is a by-product, not traced", not gas["H2"]["traced"])
    check(
        "SiH4 sticking has free-site exponent 2, inferred from site conservation",
        gas["SiH4"]["sticking"] is not None
        and gas["SiH4"]["sticking"]["freeSiteExp"] == 2,
        str(gas["SiH4"]["sticking"]),
    )

    r1, r2, r3 = ir["reactions"]
    check("R1 free-site exponent 2, inferred (input had no free sites)",
          r1["freeSiteExp"] == [2])
    check("R1 echoes the inferred free sites in its equation",
          r1["eq"] == "SiH4 + 2* -> SiH3* + H*", r1["eq"])
    check("R3 echoes the two freed sites on the right",
          r3["eq"] == "2H* -> H2 + 2*", r3["eq"])

    check("the deposited solid is named Si with its density",
          ir["solids"] == [{"name": "Si", "rho": 5.02, "rhoMaterials": {}}],
          str(ir["solids"]))
    check("R1 gas factor SiH4^1", r1["gasFactors"] == [{"idx": 0, "exp": 1}])
    check("R1 has no coverage factors", r1["covFactors"] == [])
    check("R2 coverage factor SiH3^1", r2["covFactors"] == [{"idx": 0, "exp": 1}])
    check("R3 coverage factor H^2", r3["covFactors"] == [{"idx": 1, "exp": 2}])
    check("nu rows", [r["nu"] for r in ir["reactions"]] == [[1, 1], [-1, 1], [0, -2]])
    check(
        "only R2 forms solid",
        [r["solidAtoms"] for r in ir["reactions"]] == [[0.0], [1.0], [0.0]],
    )
    return ir


def test_json_roundtrip(ir):
    print("\n2) IR survives JSON serialisation")
    text = json.dumps(ir)
    back = json.loads(text)
    t1, v1 = vc.evaluate(ir)
    t2, v2 = vc.evaluate(back)
    check("coverages identical after round-trip", all(close(a, b) for a, b in zip(t1, t2)))
    check("velocity identical after round-trip", close(v1, v2))


def test_against_reference(ir):
    print("\n3) IR evaluator vs the hand-written ViennaPS model")
    theta, v = vc.evaluate(ir)
    tS, tH = theta
    check("theta_SiH3", close(tS, REF_THETA_SIH3, rtol=1e-5), f"{tS:.6e}")
    check("theta_H", close(tH, REF_THETA_H, rtol=1e-6), f"{tH:.6e}")
    check("v_n [nm/s]", close(v, REF_V, rtol=1e-5), f"{v:.6f}")


def test_against_frontend(ir, yaml_text):
    print("\n4) IR evaluator vs the SymPy front end")
    path = write_tmp(yaml_text)
    m = vc.Mechanism(path)
    res = m.solve()
    os.unlink(path)

    theta, v = vc.evaluate(ir)
    sym = [res["theta"][c["name"]] for c in ir["coverages"]]
    for c, a, b in zip(ir["coverages"], theta, sym):
        check(f"theta_{c['name']}", close(a, b, rtol=1e-7), f"IR {a:.6e} vs SymPy {b:.6e}")
    check("v_n", close(v, res["v"], rtol=1e-7), f"IR {v:.6f} vs SymPy {res['v']:.6f}")


def test_steady_state_residual(ir):
    print("\n5) the solution satisfies the balance and the null-space ratio")
    theta, _ = vc.evaluate(ir)
    ks = vc.rate_constants(ir)
    gammas = vc.default_fluxes(ir)
    frees = vc.free_fractions(ir, theta)
    rates = [
        vc.rate(rx, ks[j], gammas, theta, frees)
        for j, rx in enumerate(ir["reactions"])
    ]
    scale = max(rates)
    for i, c in enumerate(ir["coverages"]):
        resid = sum(rx["nu"][i] * rates[j] for j, rx in enumerate(ir["reactions"]))
        check(f"residual for {c['name']} is zero", abs(resid) < 1e-9 * scale,
              f"{resid:.3e}")
    check("r1 = r2 = r3", close(rates[0], rates[1], rtol=1e-9)
          and close(rates[1], rates[2], rtol=1e-9),
          " ".join(f"{r:.6e}" for r in rates))


def test_analytic_jacobian(ir):
    print("\n6) analytic Jacobian matches a finite difference")
    ks = vc.rate_constants(ir)
    gammas = vc.default_fluxes(ir)
    site = vc.coverage_sites(ir)
    theta = [0.03, 0.11]  # an arbitrary interior point
    h = 1e-7
    ok = True
    worst = 0.0
    for j, rx in enumerate(ir["reactions"]):
        for k in range(len(theta)):
            frees = vc.free_fractions(ir, theta)
            ana = vc.drate_dtheta(rx, ks[j], gammas, theta, frees, k, site)
            tp = list(theta); tp[k] += h
            tm = list(theta); tm[k] -= h
            num = (
                vc.rate(rx, ks[j], gammas, tp, vc.free_fractions(ir, tp))
                - vc.rate(rx, ks[j], gammas, tm, vc.free_fractions(ir, tm))
            ) / (2 * h)
            denom = max(abs(ana), abs(num), 1e-30)
            worst = max(worst, abs(ana - num) / denom)
            ok &= abs(ana - num) <= 1e-5 * denom
    check("all d(rate)/d(theta) agree", ok, f"worst relative error {worst:.2e}")


def test_zero_coverage_safe(ir):
    print("\n7) derivatives are finite at zero coverage")
    ks = vc.rate_constants(ir)
    gammas = vc.default_fluxes(ir)
    site = vc.coverage_sites(ir)
    theta = [0.0, 0.0]
    frees = vc.free_fractions(ir, theta)
    ok = True
    for j, rx in enumerate(ir["reactions"]):
        for k in range(len(theta)):
            d = vc.drate_dtheta(rx, ks[j], gammas, theta, frees, k, site)
            ok &= math.isfinite(d)
    check("no division by zero", ok)


def test_eley_rideal():
    print("\n8) Eley-Rideal probe: Gamma*theta rate form and a 1:2:1 null space")
    path = write_tmp(ER)
    m = vc.Mechanism(path)
    ratios, _, _ = m.rate_ratios()
    ir = vc.build_data(m)
    res = m.solve()
    os.unlink(path)

    check("null space is 1:2:1", [int(x) for x in ratios] == [1, 2, 1],
          str([int(x) for x in ratios]))

    r2 = ir["reactions"][1]
    check("R2 couples a gas flux to a coverage",
          len(r2["gasFactors"]) == 1 and len(r2["covFactors"]) == 1,
          f"gas {r2['gasFactors']} cov {r2['covFactors']}")

    theta, v = vc.evaluate(ir)
    sym = [res["theta"][c["name"]] for c in ir["coverages"]]
    for c, a, b in zip(ir["coverages"], theta, sym):
        check(f"theta_{c['name']} matches front end", close(a, b, rtol=1e-6),
              f"IR {a:.6e} vs SymPy {b:.6e}")
    check("v_n matches front end", close(v, res["v"], rtol=1e-6),
          f"IR {v:.6e} vs SymPy {res['v']:.6e}")

    ks = vc.rate_constants(ir)
    gammas = vc.default_fluxes(ir)
    frees = vc.free_fractions(ir, theta)
    rates = [vc.rate(rx, ks[j], gammas, theta, frees)
             for j, rx in enumerate(ir["reactions"])]
    check("rates follow 1:2:1", close(rates[1], 2 * rates[0], rtol=1e-7)
          and close(rates[2], rates[0], rtol=1e-7),
          " ".join(f"{r:.6e}" for r in rates))


def test_site_density(ir):
    print("\n10) site density lives on the site type and is optional")
    check("absent when the reaction file omits it",
          ir["siteTypes"][0]["density"] is None,
          str(ir["siteTypes"][0]))
    check("no longer under constants",
          "siteDensity" not in ir["constants"], str(ir["constants"]))

    # supplied as a top-level siteTypes block
    path = write_tmp(SILANE + "\nsiteTypes:\n  - {name: default, density: 0.68}\n")
    ir_block = vc.from_file(path)
    os.unlink(path)
    check("picked up from a siteTypes block",
          close(ir_block["siteTypes"][0]["density"], 0.68),
          str(ir_block["siteTypes"][0]))

    # supplied as S0 on a surface species
    path = write_tmp(SILANE.replace("H*:    {phase: surface}",
                                    "H*:    {phase: surface, S0: 0.72}"))
    ir_s0 = vc.from_file(path)
    os.unlink(path)
    check("picked up from S0 on a surface species",
          close(ir_s0["siteTypes"][0]["density"], 0.72),
          str(ir_s0["siteTypes"][0]))

    # a supplied 0.0 must survive as 0.0 rather than collapsing to None
    path = write_tmp(SILANE + "\nsiteTypes:\n  - {name: default, density: 0.0}\n")
    ir_zero = vc.from_file(path)
    os.unlink(path)
    check("a supplied 0.0 is kept, not turned into null",
          ir_zero["siteTypes"][0]["density"] == 0.0,
          str(ir_zero["siteTypes"][0]))

    # the steady state must not depend on it
    t_none, v_none = vc.evaluate(ir)
    t_block, v_block = vc.evaluate(ir_block)
    check("steady state is independent of the site density",
          all(close(a, b) for a, b in zip(t_none, t_block)) and close(v_none, v_block),
          f"v {v_none:.6f} vs {v_block:.6f}")


def test_flux_dependence(ir):
    print("\n9) coverages respond to the local flux (as they must inside a feature)")
    lo, _ = vc.evaluate(ir, gammas=[1.0e1, 0.0])
    hi, _ = vc.evaluate(ir, gammas=[1.0e4, 0.0])
    check("theta_H increases with flux", hi[1] > lo[1],
          f"{lo[1]:.4e} -> {hi[1]:.4e}")
    check("both stay in [0,1]", all(0.0 <= t <= 1.0 for t in lo + hi))


def test_kads_sticking():
    """A sticking coefficient derived from an adsorption rate constant.

    Kinetic theory: the wall flux is (1/4) n vbar, so s = 4 k_ads / vbar with
    vbar = sqrt(8 R T / pi M). The molar mass comes from the formula.
    """
    print("\n11) sticking derived from k_ads")

    # the k_ads that reproduces s0 = 1.0 at 900 K
    c = vc.speed_coefficient("SiH4")
    k0 = 30.0 * c / 4.0
    kads_yaml = SILANE.replace(
        'type: adsorption, s0: 1.0,    Ea: 0.6',
        f'type: adsorption, k_ads: {k0!r}, Ea: 0.6')
    path = write_tmp(kads_yaml)
    ir = vc.from_file(path)
    os.unlink(path)

    check("molar mass summed from the formula (SiH4)",
          close(vc.molar_mass("SiH4"), 32.117, 1e-4),
          f"{vc.molar_mass('SiH4'):.4f} g/mol")

    st = ir["gas"][0]["sticking"]
    check("temperature exponent is -1/2", close(st["beta"], -0.5),
          str(st["beta"]))
    check("prefactor is 4 k_ads / c", close(st["s0"], 4.0 * k0 / c, 1e-12),
          f"{st['s0']:.6f}")

    # against the definition, at two temperatures
    for T in (900.0, 1200.0):
        vbar = c * math.sqrt(T)
        k_at_T = k0 * math.exp(-0.6 / (vc.KB_EV * T))
        direct = 4.0 * k_at_T / vbar
        via_ir, _ = vc.sticking(ir, "SiH4", temperature=T)
        check(f"s = 4 k_ads / vbar at {T:.0f} K", close(direct, via_ir, 1e-12),
              f"{via_ir:.6e}")

    # matches the plain-s0 mechanism where it was matched, and carries T^-1/2
    s0_ir = vc.from_file(write_tmp(SILANE))
    a, _ = vc.sticking(s0_ir, "SiH4")
    b, _ = vc.sticking(ir, "SiH4")
    check("agrees with the s0 route at 900 K", close(a, b, 1e-9),
          f"{a:.6e} vs {b:.6e}")
    a12, _ = vc.sticking(s0_ir, "SiH4", temperature=1200.0)
    b12, _ = vc.sticking(ir, "SiH4", temperature=1200.0)
    check("carries T^(-1/2) away from that point",
          close(b12 / a12, math.sqrt(900.0 / 1200.0), 1e-9),
          f"ratio {b12/a12:.6f}")

    # the steady state is reached through the same path
    theta, v = vc.evaluate(ir)
    check("growth rate matches the s0 mechanism at 900 K",
          close(v, REF_V, 1e-4), f"{v:.6f} nm/s")


def test_pressure_flux():
    """A flux derived from a partial pressure.

    Hertz-Knudsen: Gamma = p / sqrt(2 pi m kB T), equivalently (1/4) n vbar
    with n = p / (kB T). The molar mass comes from the formula, and the IR
    carries the derived flux, so nothing downstream changes.
    """
    print("\n12) flux derived from a partial pressure")

    # the pressure that reproduces Gamma = 1.0e3 (1e15 /cm^2 /s) at 900 K,
    # from the definition
    KB_J = 1.380649e-23
    N_A = 6.02214076e23
    m = vc.molar_mass("SiH4") * 1e-3 / N_A  # kg
    p = 1.0e18 * 1e4 * math.sqrt(2.0 * math.pi * m * KB_J * 900.0)  # Pa

    p_yaml = SILANE.replace("flux: 1.0e3", f"pressure: {p / 0.133322!r}")
    path = write_tmp(p_yaml); ir = vc.from_file(path); os.unlink(path)
    check("Gamma = p / sqrt(2 pi m kB T), in the default mTorr",
          close(ir["gas"][0]["flux"], 1.0e3, 1e-9),
          f"{ir['gas'][0]['flux']:.6f}")

    path = write_tmp(SILANE); ir_flux = vc.from_file(path); os.unlink(path)
    v = vc.evaluate(ir)[1]
    v_flux = vc.evaluate(ir_flux)[1]
    check("growth rate matches the flux mechanism", close(v, v_flux, 1e-8),
          f"{v:.9f} vs {v_flux:.9f} nm/s")

    # another unit is declared once, at the document level
    pa_yaml = ("pressureUnit: Pa\n" + SILANE).replace(
        "flux: 1.0e3", f"pressure: {p!r}")
    path = write_tmp(pa_yaml); ir_t = vc.from_file(path); os.unlink(path)
    check("Pa declared once at the document level",
          close(ir_t["gas"][0]["flux"], 1.0e3, 1e-9),
          f"{ir_t['gas'][0]['flux']:.6f}")

    # the shipped example is silane.yaml in reactor units
    ship = vc.from_file("silane_pressure.yaml")
    check("silane_pressure.yaml reproduces silane.yaml",
          close(ship["gas"][0]["flux"], 1.0e3, 1e-3)
          and close(vc.evaluate(ship)[1], REF_V, 1e-3),
          f"flux {ship['gas'][0]['flux']:.3f}, v {vc.evaluate(ship)[1]:.6f}")


def test_energy_units():
    """Ea may be given in kcal/mol against R; the IR is always eV.

    The two forms are identical because Ea[kcal/mol]/(R T) == Ea[eV]/(kB T),
    so the conversion happens once at the input boundary.
    """
    print("\n13) activation energy in kcal/mol")

    check("1 kcal/mol is 0.0433641 eV",
          close(vc.EV_PER_KCAL_PER_MOL, 0.0433641, 1e-5),
          f"{vc.EV_PER_KCAL_PER_MOL:.9f} eV")

    k = 1.0 / vc.EV_PER_KCAL_PER_MOL  # kcal/mol per eV
    kcal = (SILANE.replace("temperature: 900.0",
                           "temperature: 900.0\nenergyUnit: kcal/mol")
                  .replace("Ea: 0.6", f"Ea: {0.6 * k!r}")
                  .replace("Ea: 1.5", f"Ea: {1.5 * k!r}")
                  .replace("Ea: 2.0", f"Ea: {2.0 * k!r}"))

    path = write_tmp(kcal); ir_k = vc.from_file(path); os.unlink(path)
    path = write_tmp(SILANE); ir_e = vc.from_file(path); os.unlink(path)

    check("the IR stays eV and records the source unit",
          ir_k["conventions"]["energyUnit"] == "eV"
          and ir_k["conventions"]["inputEnergyUnit"] == "kcal/mol",
          str(ir_k["conventions"]["inputEnergyUnit"]))

    for a, b in zip(ir_e["reactions"], ir_k["reactions"]):
        check(f"{a['id']} Ea converts back to eV",
              close(a["constant"]["Ea"], b["constant"]["Ea"], 1e-9),
              f"{b['constant']['Ea']:.9f} eV")

    va = vc.evaluate(ir_e)[1]
    vb = vc.evaluate(ir_k)[1]
    check("steady state is unchanged by the unit", close(va, vb, 1e-9),
          f"{va:.9f} vs {vb:.9f}")

    # an unknown unit is rejected rather than silently assumed to be eV
    bad = SILANE.replace("temperature: 900.0",
                         "temperature: 900.0\nenergyUnit: joules")
    path = write_tmp(bad)
    try:
        vc.from_file(path)
        rejected = False
    except SystemExit:
        rejected = True
    os.unlink(path)
    check("an unknown energyUnit is rejected", rejected)


def test_multiple_site_types():
    """A two-site mechanism: one closure per type, and the front end (SymPy) and
    the IR evaluator must land on the same steady state."""
    print("\n14) multiple site types (two-site GaAs toy)")
    path = write_tmp(TWOSITE)
    m = vc.Mechanism(path)
    ir = vc.build_data(m)
    res = m.solve()
    os.unlink(path)

    check("two site types, A then B",
          [st["name"] for st in ir["siteTypes"]] == ["A", "B"],
          str([st["name"] for st in ir["siteTypes"]]))
    sites = {c["name"]: c["site"] for c in ir["coverages"]}
    check("Ga* on site A, As* on site B",
          sites == {"Ga*": 0, "As*": 1}, str(sites))
    fe = [rx["freeSiteExp"] for rx in ir["reactions"]]
    check("free-site exponents per type", fe == [[1, 0], [0, 1], [0, 0]], str(fe))
    check("sticking of As names its own site type B",
          ir["gas"][1]["sticking"]["site"] == 1,
          str(ir["gas"][1]["sticking"]))

    # front end (SymPy, per-type closures) vs the IR evaluator (per-type)
    th, v = vc.evaluate(ir)
    sym = [res["theta"][c["name"]] for c in ir["coverages"]]
    for c, a, b in zip(ir["coverages"], th, sym):
        check(f"theta_{c['name']} matches front end", close(a, b, rtol=1e-6),
              f"IR {a:.6e} vs SymPy {b:.6e}")
    check("v_n matches front end", close(v, res["v"], rtol=1e-6),
          f"IR {v:.6e} vs SymPy {res['v']:.6e}")

    # steady state couples the two site balances: r1 = r2 = r3
    ks = vc.rate_constants(ir)
    gam = vc.default_fluxes(ir)
    frees = vc.free_fractions(ir, th)
    rr = [vc.rate(rx, ks[j], gam, th, frees)
          for j, rx in enumerate(ir["reactions"])]
    check("r1 = r2 = r3 at steady state",
          close(rr[0], rr[1], 1e-7) and close(rr[1], rr[2], 1e-7),
          " ".join(f"{r:.4e}" for r in rr))

    # the per-type chain term: analytic Jacobian vs finite difference
    site = vc.coverage_sites(ir)
    theta = [0.2, 0.35]
    h, worst = 1e-7, 0.0
    for j, rx in enumerate(ir["reactions"]):
        for k in range(2):
            fr = vc.free_fractions(ir, theta)
            ana = vc.drate_dtheta(rx, ks[j], gam, theta, fr, k, site)
            tp = list(theta); tp[k] += h
            tm = list(theta); tm[k] -= h
            num = (vc.rate(rx, ks[j], gam, tp, vc.free_fractions(ir, tp))
                   - vc.rate(rx, ks[j], gam, tm, vc.free_fractions(ir, tm))) / (2 * h)
            denom = max(abs(ana), abs(num), 1e-30)
            worst = max(worst, abs(ana - num) / denom)
    check("per-type analytic Jacobian matches finite difference", worst < 1e-5,
          f"worst relative error {worst:.2e}")


REVERSIBLE = """
name: langmuir_probe
temperature: 900.0
species:
  A:  {phase: gas, flux: 5.0e2, atoms: {Si: 1}}
  A*: {phase: surface, atoms: {Si: 1}}
  S:  {phase: solid, rho: 5.0, atoms: {Si: 1}}
surface:
  - {eq: "A <-> A*", type: adsorption, s0: 0.5, Ea: 0.2, k0_rev: 1.0e13, Ea_rev: 1.4}
  - {eq: "A* -> S",  type: surface,    k0: 1.0e11, Ea: 1.6}
"""


def test_reversible_steps():
    """A reversible step, against the Langmuir closed form.

    For  A + * <-> A*  followed by  A* -> S + *,  the coverage balance is
        s*Gamma*(1-theta) = (k_des + k_dep)*theta
    so the steady state is exact:
        theta = s*Gamma / (s*Gamma + k_des + k_dep).
    """
    print("\n15) reversible steps (Langmuir closed form)")
    path = write_tmp(REVERSIBLE)
    m = vc.Mechanism(path)
    ir = vc.build_data(m)
    os.unlink(path)

    check("the reversible entry expanded into a forward and a reverse",
          len(ir["reactions"]) == 3, str([r["id"] for r in ir["reactions"]]))
    fwd, rev, dep = ir["reactions"]
    check("the reverse is named after its partner",
          rev["id"] == "R1r" and rev.get("reverseOf") == "R1", rev["id"])
    check("the reverse mirrors the forward equation",
          fwd["eq"] == "A + * -> A*" and rev["eq"] == "A* -> A + *",
          f"{fwd['eq']}  |  {rev['eq']}")
    check("the reverse negates the stoichiometry",
          rev["nu"] == [-x for x in fwd["nu"]], f"{fwd['nu']} vs {rev['nu']}")
    check("the reverse is a thermal constant, so it does not become a sticking",
          rev["constant"]["kind"] == "arrhenius"
          and ir["gas"][0]["sticking"]["reaction"] == "R1",
          rev["constant"]["kind"])
    check("the reverse consumes the coverage, with no free-site factor",
          rev["covFactors"] == [{"idx": 0, "exp": 1}] and rev["freeSiteExp"] == [0])

    # against the closed form
    ks = vc.rate_constants(ir)
    gam = vc.default_fluxes(ir)
    s, n = vc.sticking(ir, "A")
    sGamma = s * gam[0]
    theta_exact = sGamma / (sGamma + ks[1] + ks[2])
    theta, v = vc.evaluate(ir)
    check("theta matches the Langmuir closed form",
          close(theta[0], theta_exact, 1e-9),
          f"solved {theta[0]:.9e} vs exact {theta_exact:.9e}")
    check("growth rate matches k_dep * theta / rho",
          close(v, ks[2] * theta[0] / ir["solids"][0]["rho"], 1e-9), f"{v:.6e}")

    # forward and reverse are separate columns of nu, so the null space is no
    # longer unique; the front end falls back to the full Newton solve
    ratios, _, _ = m.rate_ratios()
    res = m.solve()
    check("the front end still solves it", close(res["theta"]["A*"], theta_exact, 1e-6),
          f"SymPy {res['theta']['A*']:.6e}")
    check("the null space is no longer 1-D, as expected for a reversible pair",
          ratios is None, "unique" if ratios is not None else "not unique")

    # a vanishing reverse constant must reproduce the irreversible mechanism
    slow = REVERSIBLE.replace("k0_rev: 1.0e13", "k0_rev: 1.0e-30")
    path = write_tmp(slow); ir_slow = vc.from_file(path); os.unlink(path)
    irrev = REVERSIBLE.replace(
        '- {eq: "A <-> A*", type: adsorption, s0: 0.5, Ea: 0.2, k0_rev: 1.0e13, Ea_rev: 1.4}',
        '- {eq: "A -> A*", type: adsorption, s0: 0.5, Ea: 0.2}')
    path = write_tmp(irrev); ir_irrev = vc.from_file(path); os.unlink(path)
    v_slow = vc.evaluate(ir_slow)[1]
    v_irrev = vc.evaluate(ir_irrev)[1]
    check("a vanishing reverse constant reproduces the irreversible mechanism",
          close(v_slow, v_irrev, 1e-9), f"{v_slow:.9f} vs {v_irrev:.9f}")


def main():
    print("Phase 1 gate: Mechanism IR")
    ir = test_silane_ir_structure()
    test_json_roundtrip(ir)
    test_against_reference(ir)
    test_against_frontend(ir, SILANE)
    test_steady_state_residual(ir)
    test_analytic_jacobian(ir)
    test_zero_coverage_safe(ir)
    test_eley_rideal()
    test_site_density(ir)
    test_flux_dependence(ir)
    test_kads_sticking()
    test_pressure_flux()
    test_energy_units()
    test_multiple_site_types()
    test_reversible_steps()

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
