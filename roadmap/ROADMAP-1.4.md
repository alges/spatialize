# Roadmap for Spatialize 1.4

> **Status:** v0.2 (2026-10-20). Development branch only (see `README.md`): not part of the 1.3 release.
> Items moved from release 1.3 to 1.4, and the experimental features of `spatialize.futures` as
> candidates to become stable. Nothing here is designed or approved for code yet: each item lists its
> evidence, what is open and the decisions to take first.
> **Theory:** the book *A General Theory of Higher-Order Geostatistics* (Egaña, Díaz, Navarro, Ehrenfeld) is
> a draft; it is cited by content, its section numbers only as dated pointers.
> **Standing rules:** a feature leaves `spatialize.futures` only when mature (studies, unit tests,
> documentation, conformance-scenario evidence); no performance regressions.

The items:

1. The widening of the local laws (calibration of the target, its shape, its spatial structure).
2. Two planned checks of scenario S03 (ordinary kriging with the true covariance; coverage of widened
   intervals).
3. The strength of the coupling of drawing decoders (ρ between `coupling="location"` and `"cell"`).
4. Multivariate ESI.
5. The experimental features of `spatialize.futures`, candidates to become stable in 1.4.

Items 1 and 3 share a question, how the law at a location and the joint law over a region are calibrated
at once; they should be designed together. Item 2's second check depends on item 1.

---

## 1. The widening of the local laws

### 1.1 What exists (1.3)

- `spatialize.empirical`: `FittedModelFactory(widening=False|"gamma"|"skew_normal"|"auto", widening_knn=12)`;
  `_local_target_variance` (variance of the 12 nearest data), `_local_target_skewness`; components
  `_gamma_widen_component`, `_skew_normal_widen_component` (a mixture of laws centred on each member,
  variance `need = τ² − s²`, book Def. 12.3.2, draft); `"auto"` chooses the family once per variable by
  the sign of the data (fixed 2026-10-16).
- `spatialize.gs.ess.widen_members` (new in 1.3, e969f52): the same components, one draw per member, the
  member index kept (`empirical._widen_members`), so regional readings keep the members' dependence.
- Users: `ess_sample`, `PosteriorAudit` (laws for PIT/flags), `law_cdf` readings of the scenarios,
  `widen_members`.

### 1.2 Evidence gathered in 1.3 (all on pinned scenario fields; numbers are means over fields)

Three distinct symptoms, probably with distinct causes.

**(a) Marginal tails too wide.**
- S10 (centred lognormal block-mark field, skew-normal widening, members widened member by member): the
  level minimising the realised asymmetric loss is pulled towards the median, 0.31 against the theory's
  α* = 1/6 at c = 5 and 0.76 against 5/6 at c = 1/5 (standard deviations near 0.02 over fields), so the
  checks `level-near-5`, `level-near-0.2` are KNOWN failures. At c = 1 the level is 0.515 (fine).
- S07 study: 90 % coverage of the widened block law 0.94 in S10's setting (over-covering), against 0.87
  in S09's (fine).
- Notebook 09 (examples): pointwise coverage of widened skew-normal laws 90.4 % at alpha 0.95 and 0.8 —
  the centre of the law is calibrated; the extremes are where it fails.
- Likely cause (hypothesis, to test): the target τ² is the variance of the 12 nearest data, which on a
  block-mark field spans several cells and so measures between-cell variance; the theory's target is the
  residual law at the location (book Prop. 12.3.3, draft: local laws are narrower than the mark law by
  the variance of the local means).

**(b) The shape of the Gamma family on heavy tails.**
- S05 (Pareto marks, tail index 1.05): on Gamma-widened members the median falls behind the mean for
  adaptive IDW (−0.25 sd per field) and kriging within the cell (−0.33), reversing the theory's claim,
  and the widened medians fall below the smallest datum in 12–38 % of the ring around the top decile
  (a Gamma's median lies below its mean). The widening raised the median's error under every decoder.
  The scenario reads raw members (spec: "members"); the reversal is recorded in its history.

**(c) Spatial structure: independent noise per location.**
- S03 v14 (V3 on simulated fields): widened simulated fields keep 0.39 (adaptive IDW) and 0.19 (IDW) of
  the truth's anisotropy coherence, against 0.73 and 0.48 unwidened; recorded, no check.
- S13: simulated fields with widening depart from the truth's Euler curve by +18 (speckle), +6.6
  without.
- S07: members widened member by member cover the true exceeded area in 11 of 24 fields (raw IDW
  members 1 of 12, the weighted draw coupled by cell 23 of 24): independent noise per location cancels
  in a regional average, so widening does not widen regional laws.
- The theory is consistent with (c): a simulated field draws from the product of the local laws (book
  §12.3.6, draft), so the noise of the widening is independent across locations by construction.

### 1.3 Directions to study (none chosen)

1. **Target by cells instead of neighbours**: τ² read from the data sharing the location's cell, over the
   partitions (a partition-aware target, consistent with the block-mark residual law), or the k nearest
   data restricted to the cell.
2. **Target calibrated by leave-one-out**: choose `widening_knn` (or a scale of τ²) so the LOO PIT of the
   data is uniform — the posterior audit already computes these PITs (`PosteriorAudit.calibration`).
3. **Shape**: a family whose median is not dragged below the data on heavy tails (lognormal components,
   or a mixture of the members with a scale, or widening in a log-ratio space for positive variables).
4. **Spatially structured widening**: noise shared within a cell for each member (one draw per tree and
   cell, as `coupling="cell"` does for draws), so regional readings widen too; or noise correlated by the
   co-occurrence of the partitions. Ties directly to item 3.
5. **Keep the pointwise calibration**: any change must preserve notebook 09's 90 % pointwise coverage and
   P4's validity (a law at every location).

### 1.4 Validation (pre-register before code)

- S10's known failures become passing checks (optimal level within ±0.05 of α*), or are explained.
- S05: a widened reading on which the median beats the mean under all four decoders.
- S03: V3 on widened simulated fields within the members' margins.
- S07: widened members covering the exceeded area at the nominal rate (a coverage-family check).
- New: LOO PIT uniformity of the widened laws on S01/S04/S08 fields (identity of a calibration curve).
- Timing against 1.3's `ess_sample` on the same fields (no regression).

### 1.5 Decisions to take

- Which symptom first ((a) calibration is the most general).
- Whether the default `widening_knn=12` may change (results change: Results-that-change section of
  the release notes).
- Whether spatially structured widening becomes an option of `ess_sample`, of `widen_members`, or both.

---

## 2. Two planned checks of scenario S03

S03 (anisotropic stationary Gaussian field, 80 pinned fields) lists them under **Planned checks** in
`docs/source/scenarios/catalog_geostatistical.rst`.

### 2.1 Ordinary kriging with the true covariance

- **Claim:** ordinary kriging with the true (anisotropic) covariance, the optimal linear predictor up to
  the unknown mean, has a lower RMSE than every ensemble estimator (paired relation over the fields).
- **Needs:** an anisotropic ordinary kriging in NumPy. 1.3 has `fields.simple_kriging_exponential`
  (anisotropic, known zero mean; the V5/V9 reference) and `fields.ordinary_kriging_exponential`
  (isotropic, S05/S06/V2). The extension is small: OK with `_aniso_matrix(a1, a2, theta)` as SK uses.
- **Expected:** a pass for every decoder (the adaptive maps reach coherence ≈ the truth's but not the BLP's
  error). Worth recording: by how much each decoder trails it (the price of not knowing the covariance,
  in the theory's terms the encoder and decoder errors).
- **Sizes:** paired t over the 80 pinned fields; effect sizes from a study before the first run.
- **Alternative to decide:** use simple kriging with the true covariance (already computed by the
  evaluator) instead of ordinary kriging; SK is the BLP when the mean is known (zero here).

### 2.2 Coverage of widened intervals

- **Claim:** the 90 % intervals of the widened laws cover at least as often as the raw members'
  intervals (paired relation on the share of the grid covered).
- **Depends on item 1**: with the 1.3 widening the result is predictable (widened laws are wider) but its
  calibration is the open question; better pre-registered as "widened intervals cover at the nominal
  rate" (coverage family, per field share of the grid) once item 1 settles the target.
- **Needs:** the widened readings in `map_visual` (through `runner.widened_members` or `law_cdf`).

---

## 3. The strength of the coupling of drawing decoders (ρ)

### 3.1 What exists (1.3)

- `coupling="location"` (default; each location draws its own uniform level) and `"cell"` (one level per
  tree and cell, read on each location's weighted law; data ordered by value) for every drawing decoder
  (`draw`, `wdraw_idw`, `wdraw_kriging`, `wdraw_adaptiveidw`, `wdraw_sharpidw`), committed a319df5
  (2026-10-16). Same law at each location in both modes; no time cost.
- Evidence for `"cell"` in 1.3's scenarios: S07, the weighted draw coupled by cell covers the true
  exceeded area in every field (48/48, 96/96 — over-covering), averaged IDW members 44–48 %; S13, members
  coupled by cell keep the Euler curve of elongated bodies (departure ≈ 1.0 on 3.7), the map of means
  departs ≈ 3 more.

### 3.2 The dropped prototype

- A Gaussian copula of strength ρ ∈ [0, 1] between the location's own level and the cell's
  (`level = Φ(√ρ·z_cell + √(1−ρ)·z_loc)`), ρ = 0 the location coupling, ρ = 1 the cell coupling.
- Patch and scripts in `roadmap/patches/coupling-strength-rho/`: `coupling-strength-rho.patch` (C++
  `include/spatialize/decoders/draw.hpp`: Acklam's normal quantile, `DrawDecoder(..., float rho)`; not
  applied), `measure.py` (area, curve, blocks on a Voronoi block-mark field), `pit16.py` (PIT of the area
  over 40 fields), `sweep.py` (ρ sweep), `timing.py`.
- Measurements (2026-10-16, notebook-16 setting): full `"cell"` overshoots the area (sd of error/spread
  0.45 against 1.05 for location); the area calibrates near ρ ≈ 0.25 (partly offsetting a bias of +0.46
  sd), the selection curve needs ρ ≥ 0.75, block averages are insensitive (0.71 → 0.75 coverage). The gap
  for blocks is the narrow law at each location (U-shaped PIT), not the coupling.
- The prototype was dropped for 1.3, to be revisited.

### 3.3 Open questions

- No single ρ calibrates area, curve and blocks: is ρ a property of the field (then estimable) or of the
  question (then a reading-time choice, which a decoder parameter cannot be)?
- Choosing ρ without the truth: cross-validate the joint law of pairs of data in one cell (the idea
  recorded with the prototype), or calibrate on the PIT of regional LOO readings.
- Interaction with item 1: spatially structured widening and ρ address the same regional spread; design
  them together so they do not double-count.
- S07's over-coverage at ρ = 1 (48/48) is a concrete target for a calibrated ρ.

### 3.4 Validation

- S07: coverage at the nominal rate (two-sided) for the calibrated ρ, against the one-sided check of 1.3.
- S13: the Euler-curve equivalence kept.
- A new scenario on a selection curve with a band (S08 setting): curve coverage.
- P2/P3 identities (mean of a draw = its averaging counterpart) unchanged for every ρ (almost sure).

### 3.5 Decisions to take

- Parameter or reading: a decoder parameter `coupling_strength`, a session setting, or a reading-time
  resampling of members.
- Whether `"cell"` stays as is (ρ = 1) with the new parameter defaulting to it.

---

## 4. Multivariate ESI

### 4.1 Where it stands

- A catalogue of every idea (36 items in 9 sections: theory, decoders for vectors, encoders, readings,
  simulation and posterior analysis, engine and API, validation, decisions) was written on 2026-10-09
  with NO design chosen and no code; this section summarises its strongest candidates.
- What exists: COESI (`spatialize.futures.coesi`, two stages, marginal ensembles at auxiliary locations
  then a joint Mondrian ensemble), kept in 1.3 with its known limits (no derivation from the theory; the
  first stage's uncertainty aggregated away; Mondrian only).
- The theory's reach includes it: several variables over one partition is "independent replicates of
  variables sharing a latent grouping" (book, last chapter, draft).

### 4.2 The strongest candidates for a first version (to confirm before any design)

- **Vector draw decoders on isotopic data** (prompt §3 idea 9): `draw`/`wdraw_*` return a whole data
  row, so every member is an observed vector; nonlinear relations, ratios, compositional closure and
  detection limits are kept exactly, with no cokriging and no cross-variogram. Largest gain, least
  theory risk; the 1.3 coupling and widening machinery apply per member.
- **Shared partition** (§2 ideas 1, 3, 4): cross-dependence through co-occurrence and vector marks;
  the scalar identities (single-point law, joint law over a query set) extend with vector atoms.
- **Values `(n, p)` in the engine** (§7 ideas 26–29): `run` returning `(q, T, p)`, decoders declaring
  their output width, `ESIResult` with vector members.

### 4.3 Validation to pre-register

- T1/T2: cross co-occurrence on a shared partition (almost sure); vector draw support (every member a
  data row); a cross-covariance identity generalising P7.
- T3: a grade–density relation (vector draws keep it, per-variable averaging breaks it); a compositional
  field (closure); a collocated secondary; baselines cokriging and independent ESI per variable in NumPy.

### 4.4 Decisions to take

- Scope: isotopic only, vector draws first?
- Shared partition only, or separate partitions from the start?
- API: `values` 2D in `gs.esi`, a new `gs.mesi`, or `futures` first?
- COESI: absorb as a target-with-covariates decoder, keep, or retire?
- Which identities go into the book before the code, so the scenarios have targets.

---

## 5. The experimental features of `spatialize.futures`

Everything in `spatialize.futures` is a candidate to become stable in 1.4. A feature leaves it when its
theory and design are settled and it carries the evidence of a stable feature: studies, unit tests, a
theory page, a reference page and a conformance scenario. For each, what exists and what promotion needs.

### 5.1 Spatial entropy and mutual information (`spatialize.futures.esmi`)

- **Exists:** `SpatialEntropy` (from the members through random partitions of their range,
  `method="partition"`, or from the widened laws of ensemble simulation, `method="ess"`) and
  `SpatialMutualInformation` (`method="colocated"`, the default since 2026-10-17: pairs of a datum and the
  other variable predicted there, weighted by the share of partitions sharing a cell with the target, minus
  a permutation bias; `method="members"`, kept). Theory: `docs/source/theory/other.rst`; examples notebook
  28 (entropy level on a known field; mutual information monotone and through a square, Spearman 0.92 and
  0.89).
- **Promotion needs:** a conformance scenario (entropy against a closed form on a field with known laws;
  mutual information near 0 for independent variables, ordered for links of known strength, a non-monotone
  link seen); a decision on the default `method` of each; the entropy through `"ess"` depends on the
  widening's calibration (item 1).

### 5.2 Co-estimation (`spatialize.futures.coesi`)

- **Exists:** two stages of ensembles (marginal ensembles at auxiliary locations, then a joint Mondrian
  ensemble), with known limits listed in its module documentation: no derivation from the theory, the
  first stage's uncertainty aggregated away, auxiliary locations uniform, Mondrian only and outside the
  session settings, marginal cross-validation only.
- **Promotion needs:** more than evidence. Its natural future is item 4: absorbed as a target-with-
  covariates decoder on the catalogue engine (one stage, the uncertainty kept), or kept as the external-
  covariate path with the difference documented, or retired. To decide with item 4.

### 5.3 Variogram inference (`spatialize.futures.variogram`, new in 1.3)

- **Exists:** `infer_variogram` (Pareto search of adaptive IDW, the map's structure tensor for the
  directions, `wdraw_adaptiveidw` members for the correlation lengths; 2D and 3D), `read_variogram` and
  `ESIResult.variogram()`; result `VariogramInference` (directions, coherence, lengths with intervals over
  the members, practical ranges, sill). Theory: `docs/source/theory/other.rst`; scenario S14; examples
  notebook 23.
- **Evidence (2026-10-20, against the field itself on the same grid):** 2D, direction within 1.4° (the
  classical directional fit 5.6°), lengths 0.92 and 1.15 times the field's; Walker Lake and Sundarbans with
  1500 data, directions within 1–3°, lengths as the field's; 3D, major direction within about 6° in 7 of 8
  cases, lengths rough (0.4 to 1.7 times the model). Scenario S14: direction and major length pass, the
  minor length 1.34 times the field's (a known failure: lengths near the spacing of the data come out long);
  version 2 added two checks of a classical nugget, removed in version 3 with that fit.
- **The nugget (2026-10-20).** The theory has no nugget of the field (block-mark covariance κ₂e continuous at
  the origin); the part a nugget plays falls to the residual weight 1 − Σwᵢ, a property of the location and
  its data, which 1.3 reports at every node (`VariogramInference.residual_weight`). A measurement error of the
  data is bounded from above in 1.3 (`error_bound`): the pooled within-cell variance of the data,
  W(λ) = τ² + (field variance inside the cells), from the order-two estimator (10.2.1) at coincident locations
  and Prop. 5.6.1 (error on the diagonal only), its minimum over a ladder of granularities. The plateau τ²
  (Thm 7.4.4) was never reached with 400 data: W at the finest usable granularity 0.09–0.15 of the variance
  on clean data, 0.29–0.43 with an error of 0.3 (S14 fields); the ladder stops once fewer than 15 % of the
  data share a cell (cost), leaving about 0.21–0.24 on clean data. Turning the bound into an estimate needs
  either more data per cell or a theory of the rate at which W(λ) approaches τ². Studied and rejected, all on 4 fields × error shares 0 / 0.1 / 0.3 and on S14's
  fields: four classical fits of the data's semivariogram in the distance the ensemble rescales (with the
  members' lengths fixed: 0.20 for 0.3, lengths shrinking to 0.73×; with the scale free: nugget, sill and
  scale trading off, right as c₀/(c₀+c₁) on short ranges and as c₀/σ² on long ones, never both; iterating
  nugget and lengths: diverging on some fields); the theory's own variogram ½E[(zᵢ−zⱼ)²] = τ² + κ₂(1 − eᵢⱼ)
  (Prop. 8.3.1, draft) with e from the ensemble's cells, whose τ² depends on the granularity (0.13 to 0.53 for
  0.3), the identity holding only when the field follows that partition law; the granularity whose
  co-occurrence fits the pairs best (reduces to the classical exponential fit, Mondrian e = exp(−λ‖h‖₁): 0.23 /
  0.39 / 0.46 for 0 / 0.1 / 0.3); the error read from the members' correlation extrapolated to the origin (an
  offset of 0.1–0.15 from the data spacing; rejected as an argument resting on the curve's shape).
- **Promotion needs:** the measurement error (above), the bias of the shortest length near the data spacing
  (S14's known failure), 3D lengths (a 3D scenario), models other than the exponential (the e⁻¹
  length is model-specific), fields with a trend (lengths then only lower bounds).

## Changelog
- v0.1 (2026-10-20): created; evidence from the 1.3 scenario work (S03 v14, S05, S07, S09, S10, S13), the
  coupling prototype and the catalogue of multivariate ideas.
- v0.2 (2026-10-20): moved into the repository (development branch only); section 5, the experimental
  features of `spatialize.futures` as candidates for 1.4.
