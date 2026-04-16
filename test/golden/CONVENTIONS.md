# Golden-Master Conventions (authoritative)

Pinned by `TGR-bhs5.1`. This document is ground truth for the cross-platform
golden master suite. If xAct and TensorGR disagree on any convention listed
here, the probe in `TGR-bhs5.2` detects it; do NOT paper over a disagreement
by adjusting this document — fix the core-module bug, per HANDOFF rule 6.

## Manifold

- **Dimension**: 4 (default for the probe and all Phase-1..3 cases).
  Phase 4 perturbation cases may use higher generality.
- **Signature**: Lorentzian mostly-plus `(−, +, +, +)`.
  - xAct: `DefMetric[-1, g[-a, -b], CD]` (the leading `-1` is the sign of the
    determinant's sign choice; xAct uses it to set the signature tag).
  - TensorGR: default `lorentzian(dim)` from `src/gr/metric.jl:21`.
- **Volume-element sign**: `√|det g|` (positive). Epsilon tensor orientation
  follows xAct's default (`epsilong`) — revisit when exterior-calculus
  goldens appear.

## Covariant derivative

- Both systems use a metric-compatible, torsion-free Levi-Civita connection.
- **Metric compatibility** (probe case 1): `∇_c g_{ab} = 0`.
  - xAct: `CD[-c][g[-a, -b]] // ToCanonical` ⟶ `0`
  - TensorGR: `simplify(TDeriv(down(:c), Tensor(:g, [down(:a), down(:b)]), :D))` ⟶ `0`
- **Torsion-free**: `∇_a ∇_b f = ∇_b ∇_a f` for scalar `f`.

## Riemann tensor

- Four-index form `R_{abcd}` with full Riemann symmetry group:
  pair-antisymmetry in `(ab)` and `(cd)`, pair-exchange `R_{abcd} = R_{cdab}`,
  first (algebraic) Bianchi `R_{a[bcd]} = 0` (multi-term, NOT captured by
  monoterm canonicalization in either system).
- Both registries use the identifier `Riem` (TensorGR) / `RiemannCD` (xAct).
- **Sign convention**: chosen so that contraction defining Ricci (below) yields
  a strictly positive scalar curvature on the round sphere. Both systems
  agree on this sign by construction; the contraction identity in Phase 0
  probe is the gate.

## Ricci tensor

- **xAct contraction**: `RicciCD[-a, -b] = R^c{}_{a c b}` — trace of
  `RiemannCD` on slots 1 (up) and 3 (down), equivalent under pair-swap to
  trace on slots 2 and 4.
- **TensorGR contraction**: `Ric_{ac} = R^b{}_{a b c}` — per
  `src/gr/curvature.jl:73-82`, trace of `Riem` on slots 1 and 3 with the
  first slot raised.
- **Identity** (probe case 2, verified both sides):
  `R^c{}_{a c b} − R_{ab} = 0`.
- **Alternative contractions** (from xAct probe, documented so future cases
  don't misread xAct output):
  - `R^c{}_{a b c} = −R_{ab}` (slots 1,4).
  - `R^c{}_{c a b} = 0` (slots 1,2, antisymmetric pair).

## Ricci scalar

- **Definition**: `R = g^{ab} R_{ab}`.
- **Identity** (probe case 3): `g^{ab} R_{ab} − R = 0` — verified both sides.
- Identifier: `RicScalar` (TensorGR) / `RicciScalarCD[]` (xAct).

## Einstein tensor

- **Definition**: `G_{ab} = R_{ab} − (½) g_{ab} R`.
- **Note** (discovered during convention probe): xAct's `ToCanonical` does NOT
  automatically expand `EinsteinCD` into its definition. The probe
  `EinsteinCD[-a, -b] − (RicciCD[-a, -b] − (1/2) g[-a, -b] RicciScalarCD[])`
  does not reduce to zero without an explicit `EinsteinToRicci` rule.
  TensorGR's `einstein_expr` in `src/gr/curvature.jl:85-93` does the
  expansion inline. Phase-3 golden `TGR-bhs5.18` must either apply the
  expansion on both sides, or compare the expanded form directly.

## Weyl tensor

- **Trace-free**: `g^{ac} C_{abcd} = 0` — verified both sides.
- Identifier: `Weyl` (TensorGR) / `WeylCD` (xAct).
- In 4D, full algebraic symmetries: same as Riemann plus trace-free on all
  four single-index traces.

## Name map

xAct ⟶ neutral ⟶ TensorGR. Used by the loader/emitter in both directions.

| Neutral       | xAct               | TensorGR       | Rank  |
|---------------|--------------------|----------------|-------|
| `g`           | `g`                | `g`            | (0,2) |
| `g_inv`       | `g` (up indices)   | `g` (up)       | (2,0) |
| `delta`       | `delta`            | `δ`            | (1,1) |
| `Riem`        | `RiemannCD`        | `Riem`         | (0,4) |
| `Ric`         | `RicciCD`          | `Ric`          | (0,2) |
| `RicScalar`   | `RicciScalarCD`    | `RicScalar`    | (0,0) |
| `Ein`         | `EinsteinCD`       | `Ein`          | (0,2) |
| `Weyl`        | `WeylCD`           | `Weyl`         | (0,4) |
| `Sch`         | `SchoutenCD`       | `Sch`          | (0,2) |
| `epsilon`     | `epsilong`         | `εg`           | (0,d) |
| `CD`          | `CD`               | `D` (default)  | op    |
| `h`           | `h`                | `h`            | (0,2) |

## Index conventions

- **Position notation**: `up(:a)` / `down(:a)` in TensorGR; `a` / `-a` in xAct.
- **Default vbundle**: `Tangent` — all Phase-0..3 cases.
- **Dummy normalization** (the golden rule): after `ToCanonical`/`canonicalize`,
  both emitters MUST rewrite contracted dummy names left-to-right as
  `d1, d2, ..., dN`. Free indices keep their original names. This makes byte-
  equal JSON comparison meaningful. Encoding is documented in
  `test/golden/schema/v1.json` once that file exists.

## Perturbation (Phase 4 preview — full spec in TGR-bhs5.20)

- Metric expansion: `g_{ab} = g^{(0)}_{ab} + ε h^{(1)}_{ab} + ε² h^{(2)}_{ab}/2 + …`
- xAct: `DefMetricPerturbation[g, h, ε]`.
- TensorGR: `define_metric_perturbation!(reg, :g, :h)`.
- Order-0 expansion of `g` returns the background metric itself (HANDOFF
  critical note). Order-0 expansion of `Ric`, `RicScalar` returns the
  background curvature; `delta_riemann`/`delta_ricci` return zero at
  order 0 — never call them for background.

## Operations covered by the `op` field

Listed in order of phase introduction:

| `op`                  | TensorGR entry point                      | Phase | xAct analogue           |
|-----------------------|-------------------------------------------|-------|-------------------------|
| `canonicalize`        | `canonicalize(expr)` / `simplify(expr)`   | 1     | `ToCanonical`           |
| `to_riemann`          | `to_riemann(expr)`                        | 3     | `RicciToRiemann`        |
| `to_ricci`            | `to_ricci(expr)`                          | 3     | `RiemannToRicci`        |
| `einstein_to_ricci`   | `einstein_to_ricci(expr)`                 | 3     | `EinsteinToRicci`       |
| `delta_ricci`         | `delta_ricci(expr, h, order)`             | 4     | `Perturbed[…, n]`       |
| `expand_perturbation` | `expand_perturbation(expr, mp, order)`    | 4     | `ExpandPerturbation`    |
| `commute_covds`       | `commute_covds(expr, :D)`                 | 5     | `SortCovDs`             |

## Sanity invariants for Phase 0 probe

These three identities MUST all evaluate to `0` on both sides before any
later phase proceeds:

1. `∇_c g_{ab} = 0` (metric compatibility)
2. `R^c{}_{a c b} − R_{ab} = 0` (Ricci contraction)
3. `g^{ab} R_{ab} − R = 0` (Ricci scalar trace)

These are the minimum to trust that TensorGR and xAct are talking about the
same objects. A fourth (`G_{ab} = R_{ab} − (½) g_{ab} R`) is NOT in Phase 0
because xAct needs an explicit `EinsteinToRicci` rule (noted above).
