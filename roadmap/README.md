# Roadmap (development branch only)

This folder holds the plans for the next versions of Spatialize: what is planned, the evidence behind
each item, what is still open and the decisions to take before any code.

It lives on `develop` and is **not part of an official release**:

- it is outside the package (`src/`) and outside the documentation (`docs/source/`), and `MANIFEST.in`
  does not include it, so neither the wheels nor the source distribution carry it;
- when a release branch `release-vX.Y.Z` is cut from `develop`, remove this folder on that branch
  (`git rm -r roadmap`) before the pull request to `main`, so `main` never holds it. The workflow
  `test-all.yaml` fails a pull request to `main` that still carries the folder.

Files:

- `ROADMAP-1.4.md` — the items planned for version 1.4;
- `patches/` — prototypes kept for those items (not applied).
