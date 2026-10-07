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
`cases.py` has 32 cases on two small Voronoi block-mark fields (one lognormal mark per Voronoi
cell): 80 samples, a 15×15 grid in 2D and 60 queries in 3D, 12 cells. Inputs are stored inside each
snapshot, so the test does not depend on numpy's RNG.

- The 24 ensemble cases (every decoder on Mondrian and Voronoi, estimation, LOO and k-fold) call
  `libspatialize.run`. They keep the names and the snapshots of the dedicated entry points they
  replaced (`estimation_esi_idw`, ...; removed 2026-10-07), which `run` reproduced bit for bit.
- The custom decoder is the cell mean written in Python; the adaptive decoder runs on one thread
  (`num_threads=1`; the same numbers as with several).
- The other cases call the remaining exported functions: the partitions, plain IDW and
  co-estimation.

`public_cases.py` adds 20 cases named `api.*` on the public Python API above the extension
(`esi_*`, the hyperparameter searches, Pareto, SPA, spatial entropy, cat_esi, plain IDW), every
seed explicit. They pin the facade between Python and C++ (`spatialize.gs`), so a redesign of the
facade can be shown to leave every result unchanged. They import the package from this checkout's
`src/python`, never an installed one.

## Usage (macOS, conda Python)
```bash
python setup.py build_ext --inplace             # header edits are tracked since 2026-10-06
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python -m pytest -q tests/refactor_guard
# (re)generate, only on pre-refactor code:
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python tests/refactor_guard/make_snapshots.py [--only CASE ...]
```
`DYLD_LIBRARY_PATH` makes the in-place build load Homebrew's libomp instead of conda's older one.
The guard refuses to run against a binary older than any `.cpp`/`.hpp` source.

Developer documentation: `docs/source/development/testing.rst`.

## Verified (2026-10-05, darwin-arm64)
- deterministic: 32/32 pass in three fresh processes;
- sensitive: adding 1e-7 to the IDW weight in `esi_idw.hpp` (now `decoders/idw.hpp`) makes exactly `estimation_esi_idw`
  fail (the LOO/k-fold IDW paths use a different kernel, 1/(1+d^p)), and reverting restores 32/32.
- used for the encoder/decoder refactor (2026-10-05): 32/32 bitwise before and after.
