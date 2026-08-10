# The by-hand silane model

`psSilaneDeposition.hpp` is the silane CVD mechanism implemented directly as a
ViennaPS model, written alongside the generator so that the generated mechanism
could be checked against what a person would have written. `silaneDeposition.cpp`
and `silaneDeposition.py` drove it.

It ships with the test suite because it is the provenance of the reference
values frozen into it:

    theta_SiH3 = 1.329904e-05      theta_H = 9.159010e-02
    v_n        = 7.177863e-02 nm/s

used by `test_ir.py` and `test_phase2.py` beside it.

The independent check on the framework is SF6/O2: `psSF6O2Etching.hpp` is
ViennaPS's own plasma model, written by someone else, and the generated
mechanism reproduces it to 0.16 % on a blanket and 0.53 % through a masked
trench (`test_sf6o2.py`).

It was removed from ViennaPS once the generated mechanism reproduced it, since
a bespoke model for one chemistry is the thing the generator replaces. To run it
again, copy the header into `ViennaPS/include/viennaps/models/`, add its
`#include` to `viennaps.hpp`, and restore the `SilaneDeposition` bindings in
`python/pyWrap.cpp` and `python/pyWrapDimension.hpp`.
