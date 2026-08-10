# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lado Filipovic, Institute for Microelectronics, TU Wien
"""Command line: compile a reaction file, or explain what was derived from it.

    python -m viennachem silane.yaml                       # to stdout
    python -m viennachem silane.yaml silane.mechanism.json # to a file
    python -m viennachem --explain silane.yaml             # the full derivation
"""

import json
import sys

from . import emit, mechanism


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # A flag is accepted in any position, and a positional that looks like one
    # is a mistake rather than a filename: `x.yaml --explain` must not quietly
    # write a file called "--explain".
    explain = False
    positional = []
    for arg in argv:
        if arg in ("-e", "--explain"):
            explain = True
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        elif arg.startswith("-"):
            sys.exit(f"unknown option '{arg}'\n{__doc__}")
        else:
            positional.append(arg)

    if not positional:
        sys.exit(__doc__)
    if len(positional) > 2:
        sys.exit(f"expected a reaction file and at most an output path, "
                 f"got {positional}")

    if explain:
        mechanism.explain(positional[0])
        return 0

    data = emit.from_file(positional[0])
    if len(positional) > 1:
        emit.write(positional[1], data)
        print(f"wrote {positional[1]}")
    else:
        print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
