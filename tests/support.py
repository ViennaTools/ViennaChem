"""Shared helpers for the test suite.

There is one route from mechanism data to a ViennaPS mechanism: the C++ reader.
`fromJSON` is that same reader, handed the data in memory rather than a path, so
nothing here is a second implementation of the builder.
"""

import json

import viennaps as ps


def build(data):
    """Mechanism data (a dict) -> ps.ChemicalMechanism."""
    return ps.ChemicalMechanism.fromJSON(json.dumps(data))
