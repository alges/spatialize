# Refactor guard (internal)

Bitwise snapshots of every function exported by the C++ extension `libspatialize`, used to prove
that a **pure refactor** (e.g. the encoder/decoder split) changes no output on the machine that
produced them.

- It is **not** an acceptance criterion and **not** part of the shared scenario suite
  (`spatialize.scenarios`, whose criteria are statistical). It is never shipped in the wheel.
- Snapshots describe the code they were taken from, **bugs included**. When a bug is fixed on
  purpose, regenerate only the affected cases (`--only <case> ...`) in the same commit as the fix.
- Snapshots are machine-specific (compiler, libm, OpenMP): they live in `snapshots/<platform>/`
  and the test is skipped on platforms without them. On a new machine, generate them from the
  commit *before* the refactor.

## Cases
`cases.py` calls the 29 legacy exported functions (Voronoi in both modes → 32 cases) on two small
Voronoi block-mark fields (one lognormal mark per Voronoi cell): 80 samples, a 15×15 grid in 2D and 60 queries in 3D,
lognormal marks, 12 cells. Inputs are stored inside each snapshot, so the test does not depend on
numpy's RNG. Custom-ESI bindings use the cell-mean decoder written in Python. The adaptive
decoder runs with `parallelize=False` (parallel runs give the same numbers since the 2026-10-05
GIL fix). The generic `libspatialize.run` is not snapshotted: it is checked against the legacy
functions, which it must reproduce bit for bit.

## Usage (macOS, conda Python)
```bash
python setup.py build_ext --inplace --force     # --force: header edits are NOT tracked by `make`
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python -m pytest -q tests/refactor_guard
# (re)generate, only on pre-refactor code:
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python tests/refactor_guard/make_snapshots.py [--only CASE ...]
```
`DYLD_LIBRARY_PATH` makes the in-place build load Homebrew's libomp instead of conda's older one.
The guard refuses to run against a binary older than any `.cpp`/`.hpp` source.

Developer documentation: `docs/source/development/testing.rst`.

## Verified (2026-10-05, darwin-arm64)
- deterministic: 32/32 pass in three fresh processes;
- sensitive: adding 1e-7 to the IDW weight in `esi_idw.hpp` makes exactly `estimation_esi_idw`
  fail (the LOO/k-fold IDW paths use a different kernel, 1/(1+d^p)), and reverting restores 32/32.
- used for the encoder/decoder refactor (2026-10-05): 32/32 bitwise before and after.
