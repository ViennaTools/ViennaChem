#!/usr/bin/env python3
"""The published format, across the language boundary and across time.

ViennaChem writes mechanism data; ViennaPS reads it in C++. That is one writer
and one reader, so there is no longer a second builder to drift from -- what can
still go wrong is the *format*:

  1. the reader mis-parses something the writer emits, and the chemistry that
     runs is not the chemistry in the file;
  2. a shipped `.mechanism.json` falls behind the `.yaml` beside it, so an
     edited reaction file silently runs as its old self;
  3. the schema changes under a reader that does not know it.

Each file is therefore compiled, handed to the C++ reader, and solved on every
material it mentions, against the reference evaluator in this library. The
evaluator uses only `math`, so agreement means the format carried everything.

Run with the ViennaPS venv:  .venv/bin/python test_loader.py
"""

import glob
import json
import os
import sys

import viennaps as ps

import viennachem as vc
from support import build

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


ps.setDimension(2)
ps.Length.setUnit("nm")
ps.Time.setUnit("s")
ps.Logger.setLogLevel(ps.LogLevel.WARNING)


def materials_in(data):
    """Every material the file names, plus the default case."""
    found = set()
    for s in data["solids"]:
        found |= set(s.get("rhoMaterials") or {})
    for rx in data["reactions"]:
        found |= set(rx.get("materialConstants") or {})
        found |= set((rx.get("ionYield") or {}).get("materials") or {})
    for g in data["gas"]:
        found |= set((g.get("sticking") or {}).get("materialConstants") or {})
    return sorted(found) + ["Si"]


def compare(path):
    """Worst RELATIVE disagreement between the C++ reader and the evaluator.

    Relative, because the two run independent damped-Newton solves and stop at
    slightly different points. On a stiff mechanism -- the full GaAs one climbs
    from a bare surface under a flux of 1e20 /cm2/s -- that shows up in the
    eighth digit, which is convergence, not a difference in what the format
    carried. A disagreement in the format shows up far larger than this.
    """
    data = vc.from_file(path)
    mech = build(data)

    worst = 0.0
    theta0 = [0.0] * len(data["coverages"])
    for material in materials_in(data):
        gamma = vc.mechanism_fluxes(data, material)
        ref_theta, ref_v = vc.evaluate(data, material=material)
        cpp_theta = mech.solveCoveragesOn(gamma, theta0, material)
        cpp_v = mech.growthRateOn(gamma, cpp_theta, material)

        def rel(a, b):
            scale = max(abs(a), abs(b))
            return abs(a - b) / scale if scale > 1e-12 else abs(a - b)

        worst = max([worst, rel(ref_v, cpp_v)]
                    + [rel(a, b) for a, b in zip(ref_theta, cpp_theta)])
    return data, worst


def shipped_is_current(path, data):
    """The `.mechanism.json` beside a reaction file must be that file, today."""
    beside = os.path.splitext(path)[0] + ".mechanism.json"
    if not os.path.exists(beside):
        return False, "missing"
    with open(beside) as fh:
        return json.load(fh) == data, "stale: recompile with `python -m viennachem`"


def main():
    print("The mechanism format, written by ViennaChem and read by ViennaPS\n")

    files = sorted(glob.glob(os.path.join(vc.REACTIONS_DIR, "*.yaml")))
    for path in files:
        name = os.path.basename(path)
        try:
            data, worst = compare(path)
        except Exception as exc:
            check(name, False, f"{type(exc).__name__}: {exc}")
            continue
        check(f"{name:<22} {len(data['reactions'])} reactions, "
              f"{len(data['coverages'])} coverages",
              worst < 1e-6, f"worst relative difference {worst:.2e}")
        current, why = shipped_is_current(path, data)
        check(f"{name:<22} shipped mechanism data is current", current,
              "" if current else why)

    print("\n  the reader refuses a format it does not know")
    stamped = vc.from_file(files[0])
    stamped["schemaVersion"] = vc.SCHEMA_VERSION + 99
    try:
        ps.ChemicalMechanism.fromJSON(json.dumps(stamped))
        check("a future schema version is rejected", False, "it was accepted")
    except Exception:
        check("a future schema version is rejected", True,
              f"version {vc.SCHEMA_VERSION + 99} refused, not guessed at")

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) failed: {FAIL}")
        sys.exit(1)
    print(f"all {len(files)} reaction files survive the round trip")


if __name__ == "__main__":
    main()
