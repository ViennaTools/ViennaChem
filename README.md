# ViennaChem

**Surface-kinetics mechanisms derived from chemical reaction equations.**

Feature-scale process simulators need a surface model: a set of coverages, the
rate laws that drive them, the steady state they settle into, and the velocity
the surface moves at. Writing one by hand means writing and compiling code for
every chemistry, and re-deriving the coverage balance and its Jacobian by hand
each time.

ViennaChem takes the chemistry as data. Given a list of surface reactions and
their rate constants, it derives the rest — the free sites, the stoichiometry,
the rate laws and their exponents, the coverage balance, its analytic Jacobian,
and the surface velocity — and emits a compact mechanism file that a solver can
consume directly. Nothing is generated, compiled or linked per chemistry.

```yaml
name: silane_polySi_LPCVD
temperature: 900.0

species:
  SiH4:  {phase: gas,     flux: 1.0e3}     # 1e15 /cm2 /s
  SiH3*: {phase: surface}
  H*:    {phase: surface}
  Si:    {phase: solid,   rho: 5.02}       # 1e22 /cm3

surface:
  - {eq: "SiH4 -> SiH3* + H*",    type: adsorption, s0: 1.0,    Ea: 0.60}
  - {eq: "SiH3* -> Si + H* + H2", type: surface,    k0: 6.8e12, Ea: 1.50}
  - {eq: "2H* -> H2",             type: desorption, k0: 6.8e12, Ea: 2.00}
```

Every empty site above is *inferred*, not written. `python -m viennachem
--explain silane.yaml` shows what was derived:

```
1) atom balance and inferred free sites, per reaction
   [R1] OK   atoms H:4=4, Si:1=1   free sites: 2* left
         inferred:  SiH4 + 2* -> SiH3* + H*
   [R3] OK   atoms H:2=2           free sites: 2* right
         inferred:  2H* -> H2 + 2*

5) mass-action rate laws (read off the reactions, not supplied)
   r1 = Gamma_SiH4*s1*theta_free_default**2
   r2 = k2*theta_SiH3
   r3 = k3*theta_H**2

7) coverage balance  dtheta/dt = nu . r = 0  (Newton, analytic Jacobian)
   theta_SiH3  = 1.329904e-05
   theta_H     = 9.159010e-02
   theta_free[default]  = 9.083966e-01

8) growth rate from the solid-forming reactions
   v_n = 0.199203187250996*k2*theta_SiH3
       = 0.071779 nm/s
```

## How it fits with ViennaPS

The two projects are coupled by one file: ViennaChem writes it, ViennaPS reads
it. Each builds and runs on its own.

```
   reaction file  (silane.yaml)          <- what you write
        |
        |  ViennaChem          SymPy, PyYAML
        v
   mechanism data (silane.mechanism.json)  <- the whole contract between them
        |
        |  ViennaPS            psChemicalMechanismIO.hpp   (a self-contained reader)
        v
   ChemicalMechanism  ->  SurfaceChemistry  ->  Process
                             |
                             +- coverages and the surface velocity, per point
                             +- one particle per gas species, plus an ion
```

ViennaChem derives the chemistry; ViennaPS supplies what the chemistry cannot
know by itself — the geometry, and the flux arriving at every point of it from
ray tracing. The mechanism turns those fluxes into coverages and a velocity; the
level set moves.

### What is a prerequisite for what

| to do this | you need |
|---|---|
| write a reaction file and compile it | ViennaChem |
| solve a mechanism analytically: coverages, blanket rate, per material | ViennaChem alone (`viennachem.evaluate` imports only `math` and `json`) |
| run a **compiled** mechanism in a simulation | ViennaPS alone — its reader parses the file itself |
| run a mechanism **from the `.yaml`** | both: ViennaChem to compile, ViennaPS to run |
| change the chemistry | ViennaChem, then recompile the `.mechanism.json` |

So a C++ user runs every shipped mechanism straight from the checkout, since the
compiled file travels with the example. ViennaChem is what you add to write a
new one.

### The pieces on the ViennaPS side

| | |
|---|---|
| `psChemicalMechanismIO.hpp` | reads the file, checks its `schemaVersion` |
| `ChemicalMechanism<T>` | the mechanism in memory: coverages, reactions, ions, solids |
| `SurfaceChemistry<T,D>` | the process model: the coverage solve, the velocity, and the particles |
| `ps.ChemicalMechanism.fromFile` / `.fromJSON` | the same reader from Python, so the format has one implementation |
| `gpu::SurfaceChemistry` + `SurfaceChemistry.cuh` | the same mechanism traced on the device |

## Installation

Python 3.9 or newer, and pip.

```bash
pip install git+https://github.com/ViennaTools/ViennaChem@main
```

or from a checkout, which is what you want to run the tests:

```bash
git clone https://github.com/ViennaTools/ViennaChem
pip install ./ViennaChem
```

That is the whole installation. SymPy and PyYAML are declared as dependencies,
so pip fetches them with the package:

```
Installing collected packages: mpmath, sympy, PyYAML, viennachem
```

You can now write reaction files, compile them and solve them — both as a module
and as a command:

```bash
python -m viennachem silane.yaml --explain
viennachem silane.yaml silane.mechanism.json
```

### To run mechanisms in a simulation

Simulating a feature — transport, coverages over a geometry, a moving surface —
is [ViennaPS](https://github.com/ViennaTools/ViennaPS)'s job. Build it with the
`SurfaceChemistry` model:

```bash
git clone https://github.com/ViennaTools/ViennaPS
python ViennaPS/python/scripts/install_ViennaPS.py --venv .venv
```

Add `--no-gpu` to build without CUDA; the CPU flux engine runs every mechanism.

## Usage

```python
import viennachem as vc

data = vc.from_file("silane.yaml")           # compile the reactions
vc.write("silane.mechanism.json", data)      # what a solver consumes

theta, v = vc.evaluate(data)                 # the reference answer
theta, v = vc.evaluate(data, material="SiGe")  # on a particular material
```

From the command line:

```bash
python -m viennachem silane.yaml                        # compiled data to stdout
python -m viennachem silane.yaml silane.mechanism.json  # to a file
python -m viennachem --explain silane.yaml              # the full derivation
```

### With ViennaPS

`ps.SurfaceChemistry` is one model that covers every chemistry:

```python
import json, viennachem as vc, viennaps as ps

mech  = ps.ChemicalMechanism.fromJSON(json.dumps(vc.from_file("sf6o2.yaml")))
model = ps.SurfaceChemistry(mech)
ps.Process(domain, model, duration).apply()
```

```cpp
auto mech = readChemicalMechanism<double>("sf6o2.mechanism.json");
SurfaceChemistry<double, 2> model(mech);
```

`ViennaPS/examples/surfaceChemistry` runs eleven mechanisms — deposition,
etching, sputtering and a passivating film — through one driver, in either
language.

## The reaction file

### Top level

| key | meaning |
|---|---|
| `name` | the mechanism's name; used for output files |
| `temperature` | surface temperature in K |
| `energyUnit` | `eV` (default) or `kcal/mol`, applied to every `Ea` |
| `species` | the species and their phases, below |
| `surface` | the reactions, below |
| `siteTypes` | optional, `[{name, density}]`, the site density per type |

### Species

A name is read as a chemical formula, so `SiH4` is one Si and four H. A trailing
`*` marks an adsorbate (`H*` is hydrogen occupying one site).

| phase | keys | meaning |
|---|---|---|
| `gas` | `flux` | incident flux in 1e15 /cm² /s |
| `surface` | `site`, `S0` | the site type it occupies (default `"default"`), and optionally that type's site density |
| `solid` | `rho` | density in 1e22 /cm³; a number, or a mapping `{default: …, Mask: …}` per material |
| `ion` | `flux`, `meanEnergy`, `sigmaEnergy`, `exponent`, `inflectAngle`, `n_l`, `minAngle`, `thetaRMin`, `thetaRMax` | the ion source; energies in eV, angles in degrees |

`atoms: {Si: 1}` states a composition explicitly, for a product whose name is not
a formula (a sputtered silicon atom leaving a surface, where `Si` is already the
solid). A product that simply leaves needs nothing but its phase; the
`reemit_only: true` seen in the shipped files is a note to the reader, not a key
the compiler acts on.

### Reactions

```yaml
- {eq: "SiH4 -> SiH3* + H*", type: adsorption, s0: 1.0, Ea: 0.60}
```

| key | meaning |
|---|---|
| `eq` | the equation. `->` or `<->` for a reversible step. Empty sites are inferred and must not be written |
| `type` | `adsorption` (the prefactor is a sticking coefficient), `sputtering` or `ion_enhanced` (the rate comes from an ion yield), anything else (`surface`, `desorption`, …) is a thermal step |
| `s0` / `k0` | the prefactor: a sticking coefficient for an adsorption, an Arrhenius prefactor otherwise |
| `k_ads` | an adsorption rate constant in cm/s, converted to a sticking coefficient by kinetic theory, `s = 4k_ads/v̄` |
| `Ea`, `beta` | `k = prefactor · T^β · exp(−Ea/k_BT)` |
| `k0_rev`, `Ea_rev`, `beta_rev` | the reverse constants of a `<->` step |
| `order` | explicit rate-law exponents, `{F*: 1}`, where a published law is not mass action. The stoichiometry is unaffected |
| `materials` | per-material constants, below |
| `yield` | `{A, Eth, B, materials}` for an ion-driven step |

### Material dependence

A `materials:` entry is either a number, which scales the prefactor, or a mapping
that overrides any of `s0`/`k0`, `Ea` and `beta`. `default` covers every material
not named, and a prefactor of zero means the step does not occur there — which is
how a reaction is restricted to one material.

```yaml
- {eq: "SiH4 -> SiH3* + H*", type: adsorption, s0: 1.0, Ea: 0.60,
   materials: {default: 0.0,                 # inert everywhere else
               SiGe: {s0: 0.80, Ea: 0.45},   # a lower barrier on SiGe
               Si:   {s0: 1.00, Ea: 0.60},
               PolySi: 1.0}}
```

A partial override inherits the rest, so `materials: {SiGe: {Ea: 1.6}}` changes
only the barrier. The same block applies to `yield`, with `A` and `Eth` per
material.

### Ions

An ion-driven step carries a yield instead of a rate constant:

```
Y = A · max(√E − √E_th, 0) · f(θ_inc)
```

which is not a mass-action term: the energy is drawn from a distribution and
decays on every reflection, and the angle is local. The solver's ion particle
evaluates it per ray and deposits a yield-weighted flux, so the chemistry reads
an ordinary flux and stays mass action.

```yaml
species:
  Ar_ion: {phase: ion, flux: 5.0e1, meanEnergy: 100.0, sigmaEnergy: 10.0}

surface:
  # ions striking four fluorinated sites remove one silicon
  - {eq: "Ar_ion + 4F* + Si -> SiF4", type: ion_enhanced, order: {F*: 1},
     yield: {A: 0.15, Eth: 10.0, materials: {Mask: {A: 0.005, Eth: 40.0}}}}
```

An ion is a projectile: it delivers energy and leaves neutralised, so it takes no
part in the atom balance.

### Etching, and more than one solid

Etching falls out of the stoichiometry. A solid on the right of an equation is
deposited; a solid on the *left* is consumed, so the surface velocity is
negative. A mechanism may move several solids — a polymer deposited while a
substrate is etched — and each contributes its own atom count over its own
density:

```
v_n = Σ_j n_(S,j) · r_j / ρ_(solid of j)
```

## What it expresses

| | |
|---|---|
| rate laws | mass action from the equation, or an explicit order where a published law is not |
| free sites | inferred from site conservation |
| site types | one closure per type, so a coverage on one does not block another |
| reversibility | `A <-> B`, expanded into a second reaction |
| rate constants | `k₀ T^β exp(−Ea/kT)`, `Ea` in eV or kcal/mol |
| sticking | given, or derived from an adsorption rate constant |
| material dependence | prefactor, barrier, exponent, sticking and ion yield, per material |
| ions | energy- and angle-dependent yields, resolved per ray by the solver |
| solids | more than one, each with its own density, itself per material |
| deposition and etching | the same mechanism; the sign follows from the stoichiometry |

Coverages are solved at steady state, and incident fluxes are supplied. On the
roadmap: transient coverages, and a gas-phase level that computes the fluxes
from a bulk composition.

## The compiled mechanism

`x.mechanism.json` carries indices and exponents, never expressions, so a
consumer with no symbolic algebra can rebuild both the residual and its Jacobian:

```json
{
  "schemaVersion": 1,
  "name": "silane_polySi_LPCVD",
  "coverages": [{"name": "SiH3*", "site": 0}, {"name": "H*", "site": 0}],
  "reactions": [
    {"id": "R3", "eq": "2H* -> H2 + 2*",
     "constant": {"kind": "arrhenius", "k0": 6.8e12, "Ea": 2.0, "beta": 0.0},
     "gasFactors": [], "covFactors": [{"idx": 1, "exp": 2}],
     "freeSiteExp": [0], "nu": [0.0, -2.0], "solidAtoms": [0.0]}
  ]
}
```

`schemaVersion` is the compatibility handle: a reader that does not know a
version must refuse the file rather than guess at it. ViennaPS's reader does.

## Validation

Every mechanism is checked at three levels — the symbolic front end, the
reference evaluator, and the C++ solver — and against external references:

| check | result |
|---|---|
| analytic Jacobian vs finite difference | 4.7e-10 (one site type), 5.9e-10 (two) |
| vs a by-hand implementation of the same silane mechanism | `θ_H = 0.09159`, `v = 0.07178` nm/s |
| **vs ViennaPS's SF₆/O₂ plasma etch model, written independently** | blanket 0.16 %, masked trench 0.53 % |
| vs the published diamond radical fraction | 0.26 % over 900–1400 K |
| ion-enhanced etching vs analytic | 0.22 %, reproducing a 15× ion–neutral synergy |
| physical sputtering vs analytic | 0.89 %, mask selectivity 15.11 vs 15.04 |
| CPU vs GPU transport | 0.03 % to 0.5 % across mechanisms |

## Tests

The suite spans the whole chain, so it needs ViennaPS as well:

```bash
cd tests && python test_ir.py
```

| test | what it gates |
|---|---|
| `test_ir.py` | 76 checks on the compiler and the emitted data; no simulation |
| `test_loader.py` | the format across the language boundary: the C++ reader against the reference evaluator on every reaction file, plus staleness and schema-version refusal |
| `test_phase2.py` | the C++ solver against the hand-written silane model |
| `test_multisite.py` | two site types, on CPU and GPU |
| `test_selective.py` | material-dependent rates, through a simulation |
| `test_ion.py` | sputtering, ion-enhanced etching, and the ion–neutral synergy |
| `test_gpu.py` | CPU against GPU, including on an ion mechanism |
| `test_sf6o2.py` | the acceptance test: SF₆/O₂ against the hand-written plasma model |
| `test_polymer.py` | two solids, passivation competing with the etch |

`test_ir.py` and `test_loader.py` are fast; the rest run simulations.

`tests/reference/` holds the by-hand implementation of the silane mechanism that
the frozen reference values come from, so every number the suite checks against
is either derived in the test, read from a file in this repository, or produced
by a ViennaPS model the test itself instantiates.

## Shipped mechanisms

Eleven reaction files travel with the package, in `viennachem/examples/`, so any
of them can be compiled or solved by name from anywhere:

```bash
python -m viennachem --explain diamond.yaml
```

They are silane LPCVD, selective silane,
a SiGe/Si stack, GaAs MOVPE, diamond CVD, argon sputtering, CF₄/Ar ion-enhanced
etching, SF₆/O₂, and a passivating polymer etch. Their sources:

- Mountziaris & Jensen, *J. Electrochem. Soc.* **138** (1991) — GaAs MOVPE
  surface mechanism, Table II
- Bristol CVD Diamond Group, [standard growth
  model](https://www.chm.bris.ac.uk/pt/diamond/growthmodel.htm) — diamond
- ViennaPS `psSF6O2Etching` — the hand-written plasma model the SF₆/O₂ file is
  checked against

Where rate constants are illustrative rather than published, the file says so.

## Related

- [ViennaPS](https://github.com/ViennaTools/ViennaPS) — process simulation; reads
  these mechanisms
- [ViennaLS](https://github.com/ViennaTools/ViennaLS),
  [ViennaRay](https://github.com/ViennaTools/ViennaRay) — the level-set and
  ray-tracing libraries beneath it

## License

GPL-3.0-or-later, the same licence as
[ViennaPS](https://github.com/ViennaTools/ViennaPS). See [LICENSE](./LICENSE).

Copyright (C) 2026 Lado Filipovic, Institute for Microelectronics, TU Wien.
