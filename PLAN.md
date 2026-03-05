# TensorGR.jl — Full xAct Replacement Plan

## Context

TensorGR.jl has a working skeleton (M1-M9, 2362 lines, 2051 tests) but covers ~25% of xAct's functionality. This plan maps every xAct subsystem to a concrete implementation phase, with research rounds for unknowns. The xAct source lives at `reference/xAct/xAct/`.

**Source of truth**: xAct 1.3.0 — xTensor (10,468 lines), xPerm (1,750), xPert (476), xCoba (4,295), xTras (5,165), xTerior (1,178), xCore (824), Spinors (1,004).

**Target**: full feature parity with xTensor + xPert + xCoba + xTras core, with xTerior and Spinors as optional modules. Estimated ~8,000–10,000 additional lines of Julia.

**Parallelism note**: Phases with no data dependencies can run as parallel subagents (e.g., Phase 4 components and Phase 6 exterior calculus are fully independent). Within a phase, steps are sequential unless noted.

---

## Phase 0: Infrastructure Hardening (prerequisite for everything)

### 0.1 — Rule system (`src/rules.jl`, ~200 lines)
Pattern-matching rewrite rules — the Julia equivalent of xAct's UpValues/DownValues.
- `struct RewriteRule{P,R}` — pattern + replacement + conditions
- `@rule lhs => rhs when cond` macro for ergonomic rule definition
- `apply_rules(expr, rules)` — single pass
- `apply_rules_fixpoint(expr, rules; maxiter=100)` — to fixed point
- `AutomaticRules` registry on TensorRegistry (rules triggered by tensor name)
- **Tests**: rule application, fixed point, conditional rules
- **xAct ref**: `xCore/xCore.m` lines 114–225 (FoldedRule, AutomaticRules)

### 0.2 — Scalar algebra engine (`src/scalar/algebra.jl`, ~400 lines)
Polynomial/rational arithmetic on `Expr` trees. Currently empty.
- `scalar_expand(ex)` — distribute `*` over `+`
- `scalar_collect(ex, var)` — collect powers of `var`
- `scalar_factor(ex)` — basic factorization (GCD-based)
- `scalar_cancel(ex)` — cancel common factors in ratios
- `scalar_subst(ex, rules)` — substitute values
- Work on `Expr` trees OR `Rational{Int}` (dispatch)
- **Tests**: polynomial identities, det M = 8(1−3β)k⁴p⁴ symbolically
- **xAct ref**: Mathematica kernel provides this; we must build it

### 0.3 — Simplification pipeline (`src/algebra/simplify.jl` rewrite, ~300 lines)
The orchestrated `simplify(expr)` that chains operations to fixed point.
- `simplify(expr; registry=current_registry())` — the user-facing function
- Pipeline: `expand_products → contract_metrics → canonicalize → collect_terms → apply_rules`
- Fixed-point loop with change detection
- Options: `UseMetricOnVBundle`, `OverDerivatives`
- **Tests**: g^{ab}g_{bc} = δ^a_c, R^a_{bad} = R_{bd}, R_{a[bcd]} = 0 (after Phase 1)
- **xAct ref**: `xTensor.m` lines 9557–9598 (ToCanonical pipeline)

---

## Phase 1: Full Canonicalization Engine (~1,500 lines)

The single most impactful phase. Makes everything downstream work.

### 1.1 — Canonicalize nested expressions (`src/algebra/canonicalize.jl` rewrite)
Current canonicalize bails on non-Tensor factors. Must handle:
- Products containing `TDeriv` (derivatives as "indexed objects")
- Products containing `TProduct` (nested products)
- Canonicalize within each term of a `TSum`
- **Algorithm**: "Implode" derivatives into compound tensors (xTensor line 9724), canonicalize, then "Explode" back
- **xAct ref**: `xTensor.m` lines 9620–9670 (ToCanonicalObject), 9724–9755 (derivative handling)

### 1.2 — Multi-term symmetry rules (`src/gr/bianchi.jl`, ~300 lines)
Bianchi identities are NOT permutation symmetries — they're rewrite rules.
- First (algebraic) Bianchi: `R_{a[bcd]} = 0` → rule `R_{abcd} + R_{acdb} + R_{adbc} → 0`
- Second (differential) Bianchi: `∇_{[e} R_{ab]cd} = 0`
- Contracted Bianchi: `∇^a G_{ab} = 0`, `∇^a R_{ab} = (1/2)∇_b R`
- Implement as `RewriteRule` objects, registered automatically by `define_curvature_tensors!`
- **Tests**: R_{a[bcd]}=0, ∇^a G_{ab}=0
- **xAct ref**: `xTensor.m` lines 7048–7092 (curvature relations), `xTras/xTensor.m` (CurvatureRelationsBianchi)

### 1.3 — Dummy index renaming during canonicalization (~200 lines)
When canonicalizing, dummies must be renamed to a canonical alphabet.
- After xperm returns the canonical permutation, rename dummies to canonical names (`a`, `b`, `c` ... from the manifold's index alphabet)
- `SameDummies(expr)` — minimize the number of distinct dummy names
- **xAct ref**: `xTensor.m` lines 3141–3180 (DummyIn, NewIndexIn)

### 1.4 — collect_terms up to symmetry equivalence (~200 lines)
Current `collect_terms` only matches structurally identical terms. Must match terms that are equal after canonicalization.
- Canonicalize each term, then collect by canonical form
- `collect_terms(expr) = canonicalize each term → group by canonical form → sum coefficients`
- **Tests**: `R_{abcd} + R_{cdab}` → `2R_{abcd}`, `T_{ab} + T_{ba}` → `2T_{ab}` (if symmetric)

### 1.5 — RESEARCH ROUND: xSort/Object intermediate representation
**Question**: Do we need an intermediate "Object" representation like xTensor's xSort?
- Read `xTensor.m` lines 9036–9294 (Object, xSort, ObjectSort)
- Decide: translate to Julia struct, or handle via dispatch on TensorExpr directly?
- **Output**: Decision document + implementation plan for Object repr if needed

---

## Phase 2: Covariant Derivatives (~1,200 lines)

### 2.1 — DefCovD and Christoffel symbols (`src/gr/covd.jl`, ~400 lines)
- `@derivative ∇ on=M4 metric=g` macro (already planned, not implemented)
- Stores: VBundles, manifold, metric compatibility, torsion/curvature flags
- Auto-generates: Christoffel[∇], Torsion[∇], Riemann[∇], Ricci[∇], etc.
- Metric compatibility: `∇_a g_{bc} = 0` as automatic rule
- **xAct ref**: `xTensor.m` lines 5972–7289 (full CovD section)

### 2.2 — ChangeCovD: connection switching (`src/gr/covd.jl` continued, ~300 lines)
- `change_covd(expr, ∇₁, ∇₂)` — replace ∇₁ with ∇₂ + Christoffel terms
- `covd_to_partial(expr, ∇)` — expand ∇ into ∂ + Γ (ChangeCovD to PD)
- Leibniz rule for connection terms on each free index
- **xAct ref**: `xTensor.m` lines 6016–6074 (ChangeCovD, makechangeCovD)

### 2.3 — SortCovDs: derivative commutation (`src/gr/sort_covds.jl`, ~300 lines)
- `sort_covds(expr, ∇)` — sort covariant derivatives into canonical order
- When commuting `∇_a ∇_b` to `∇_b ∇_a`: insert Riemann + torsion terms
- `commute_covds(expr, ∇, a, b)` — commute specific pair
- Fixed-point loop until all derivative pairs are sorted
- **Tests**: `[∇_a, ∇_b]V^c = R^c_{dab} V^d`
- **xAct ref**: `xTensor.m` lines 6549–7289 (SortCovDs, CommuteCovDs)

### 2.4 — Lie derivatives (`src/gr/lie.jl`, ~200 lines)
- `lie_d(v, expr)` — Lie derivative along vector field v
- Leibniz rule on products, specific formulas for tensors
- `£_v T^{a...}_{b...} = v^c ∇_c T - T^{c...}_{b...} ∇_c v^a + T^{a...}_{c...} ∇_b v^c - ...`
- **xAct ref**: `xTensor.m` lines 7566–7879 (Lie derivatives, brackets)

---

## Phase 3: Automatic Perturbation Theory (~1,500 lines)

Replace hard-coded δRicci/δRicciScalar with xPert's partition-based algorithm.

### 3.1 — Partition combinatorics (`src/perturbation/partitions.jl`, ~150 lines)
- `sorted_partitions(n)` — all partitions of integer n
- `all_compositions(m, n)` — all ordered compositions of m into n parts
- Multinomial coefficients
- **xAct ref**: `xPert.m` lines 135–159 (SortedPartitions, AllPartitions)

### 3.2 — DefMetricPerturbation (`src/perturbation/metric_perturbation.jl`, ~400 lines)
- `@perturb g => h with ε` — define metric perturbation
- Auto-generates expansion rules for: g⁻¹, Christoffel, Riemann, Ricci, RicciScalar, Einstein, Weyl
- Uses partition-based recursion: `δⁿΓ` derived from `δⁿg`, NOT hard-coded
- **Key formula**: `δⁿΓ^a_{bc} = Σ_{partitions} (−1)^|π| C(π) · productThreePert(π)`
- **xAct ref**: `xPert.m` lines 384–462 (DefGenPert* functions)

### 3.3 — ExpandPerturbation engine (`src/perturbation/expand.jl`, ~400 lines)
- `expand_perturbation(Perturbation(expr, n))` — expand to order n
- Dispatch: metric → inverse metric formula, Christoffel → ThreePert recursion, Riemann → derivative of Christoffel, Ricci → contracted Riemann, RicciScalar → Leibniz on Ricci×g⁻¹
- Nothing hard-coded: all derived from `∇δg`
- **Tests**: Reproduce δRicci, δRicciScalar from Phase M7 results; verify at order 2
- **xAct ref**: `xPert.m` lines 268–462 (ExpandPerturbation1 rules)

### 3.4 — Gauge transformations (`src/perturbation/gauge.jl`, ~200 lines)
- `gauge_change(pert, ξ, order)` — Bruni-Damour recursion
- `δ'¹ = δ¹ + £_ξ⁰ δ⁰`, `δ'² = δ² + £_ξ⁰ δ¹ + £_ξ¹ δ⁰ + (1/2)£²_ξ⁰ δ⁰`
- **xAct ref**: `xPert.m` lines 366–376 (GaugeChange, BruniTerm)

### 3.5 — Variational derivatives (`src/perturbation/variation.jl`, ~200 lines)
- `var_d(lagrangian, field)` — functional derivative δL/δφ
- Uses IBP to move derivatives off the variation
- **xAct ref**: `xTensor.m` lines 7879–8161 (variational derivatives)

### 3.6 — RESEARCH ROUND: Faà di Bruno formula
- xPert uses the Faà di Bruno formula for perturbation of scalar functions
- **Question**: Do we need this for TensorGR.jl? When does a user encounter `f(g(ε))`?
- **Output**: Decision on whether to implement, and how

---

## Phase 4: Component Calculations (xCoba equivalent, ~1,500 lines)

**PARALLELIZABLE**: independent of Phases 2, 3, 5, 6.

### 4.1 — Basis/Chart system (`src/components/basis.jl`, ~300 lines)
- `@chart Cartesian on=M4 coords=[t,x,y,z]` — define coordinate chart
- `@basis frame on=M4` — define non-coordinate basis
- Stores: component indices, PDOfBasis, VBundleOfBasis
- **xAct ref**: `xCoba.m` lines 302–450 (DefBasis, DefChart)

### 4.2 — CTensor: component arrays (`src/components/ctensor.jl`, ~400 lines)
- `struct CTensor` — holds Array + basis info + density weight
- Arithmetic: `+`, `*`, contraction, trace, inverse, determinant
- Basis change via Jacobian matrices
- **xAct ref**: `xCoba.m` lines 1227–1400 (CTensor operations)

### 4.3 — ToBasis / ComponentArray (`src/components/to_basis.jl`, ~300 lines)
- `to_basis(expr, chart)` — convert abstract-indexed expression to component array
- `component_array(expr, bases...)` — extract all components
- SplitIndex: replace abstract index with range of component indices
- **xAct ref**: `xCoba.m` lines 1089–1175 (ToBasis, ComponentArray)

### 4.4 — MetricCompute: automatic Christoffel/Riemann from metric (`src/components/metric_compute.jl`, ~300 lines)
- `metric_compute(g::CTensor, chart, :Christoffel)` — compute from metric components
- `metric_compute(g, chart, :Riemann)` — from Christoffel
- `metric_compute(g, chart, :Ricci)`, `:RicciScalar`, `:Einstein`, `:Weyl`
- All computed from partial derivatives, not formulas
- **Tests**: Schwarzschild metric → known Christoffel/Riemann components
- **xAct ref**: `xCoba.m` lines 3200–3480 (MetricCompute)

### 4.5 — ComponentValue storage (`src/components/values.jl`, ~200 lines)
- `component_value!(expr, value)` — store component value
- Uses canonicalization to identify independent components
- Symmetry-aware: only stores independent components
- **xAct ref**: `xCoba.m` lines 2150–2260 (ComponentValue, TensorValues)

---

## Phase 5: Curvature Tensor Algebra (xTras core, ~800 lines)

### 5.1 — Curvature conversions (`src/gr/conversions.jl`, ~300 lines)
- `riemann_to_weyl(expr)` — Riemann = Weyl + Ricci/RicciScalar decomposition
- `weyl_to_riemann(expr)`, `ricci_to_einstein(expr)`, `einstein_to_ricci(expr)`
- `ricci_to_tf_ricci(expr)` — trace-free Ricci decomposition
- `riemann_to_schouten(expr)` — Schouten tensor
- `contract_curvature(expr)` — Riemann → Ricci → RicciScalar
- **xAct ref**: `xTensor.m` lines 8161+ (curvature conversions), `xTras/xTensor.m`

### 5.2 — Tracelessness and projections (`src/algebra/trace.jl`, ~200 lines)
- `make_traceless(expr, metric)` — decompose T = T_TF + (1/d)g·tr(T)
- `trace(expr, idx1, idx2)` — contract two indices
- **xAct ref**: `xTras/Algebra.m` (MakeTraceless)

### 5.3 — Young tableaux (`src/algebra/young.jl`, ~300 lines)
- `young_symmetrize(expr, tableau)` — project onto irreducible representation
- `young_project(expr, tableau)` — Young projection
- Riemann Young decomposition: Weyl, Ricci, scalar pieces
- **xAct ref**: `xTras/Algebra.m` (YoungSymmetrize, YoungProject, RiemannYoungProject)

---

## Phase 6: Exterior Calculus (xTerior equivalent, ~800 lines)

**PARALLELIZABLE**: independent of Phases 2, 3, 4, 5.

### 6.1 — Differential forms (`src/exterior/forms.jl`, ~200 lines)
- `@form ω on=M4 degree=2` — define k-form
- Graded algebra: degree tracking
- **xAct ref**: `xTerior.m` lines 91–155

### 6.2 — Wedge product and Hodge dual (`src/exterior/operations.jl`, ~300 lines)
- `wedge(α, β)` — supercommutative graded product
- `hodge(metric, form)` — Hodge star operator
- `codiff(metric, form)` — codifferential δ = ★d★
- **xAct ref**: `xTerior.m` lines 162–295 (Wedge), lines 97–120 (Hodge)

### 6.3 — Exterior derivative and Cartan formula (`src/exterior/derivative.jl`, ~200 lines)
- `ext_d(form)` — exterior derivative
- `interior(v, form)` — interior product ι_v
- `cartan_d(v, form)` — Cartan/Lie derivative on forms = dι_v + ι_v d
- **xAct ref**: `xTerior.m` lines 111–143

### 6.4 — Connection and curvature forms (`src/exterior/connection_forms.jl`, ~100 lines)
- `connection_form(∇)`, `curvature_form(∇)`
- Cartan structure equations: dθ = −ω∧θ + T, dω = −ω∧ω + Ω
- **xAct ref**: `xTerior.m` lines 127–143

---

## Phase 7: Advanced Features (~1,000 lines)

### 7.1 — Killing vectors (`src/gr/killing.jl`, ~100 lines)
### 7.2 — Ansatz construction (`src/algebra/ansatz.jl`, ~200 lines)
### 7.3 — CollectTensors (`src/algebra/collect_tensors.jl`, ~200 lines)
### 7.4 — Symmetric covariant derivatives (`src/gr/sym_covds.jl`, ~200 lines)
### 7.5 — Index-free notation (`src/notation/index_free.jl`, ~150 lines)
### 7.6 — RESEARCH ROUND: Invar database integration

---

## Phase 8: Package Extensions and Distribution (~500 lines)

### 8.1 — Symbolics.jl extension
### 8.2 — SymEngine.jl extension
### 8.3 — BinaryBuilder for xperm.c
### 8.4 — Documentation via Documenter.jl

---

## Execution Order & Parallelism

```
Phase 0 (infrastructure) ──→ Phase 1 (canonicalization) ──→ Phase 2 (covd)
                                                                  │
Phase 3 (perturbation) ←──────────────────────────────────────────┘
         │
         ├──→ Phase 4 (components)     [PARALLEL — independent]
         ├──→ Phase 5 (curvature alg)  [needs Phase 1+2]
         └──→ Phase 6 (exterior calc)  [PARALLEL — independent]

Phase 7 (advanced) ←── all above
Phase 8 (distribution) ←── all above
```

**Parallelizable pairs** (can be separate subagents / worktrees):
- Phase 4 (components) ∥ Phase 6 (exterior calculus) — zero shared code
- Phase 5.2 (trace) ∥ Phase 5.3 (Young) — independent algorithms
- Phase 7.1-7.5 — all independent of each other
- Phase 8.1 ∥ 8.2 ∥ 8.3 — independent extensions

**Sessions estimate** (at ~2000 lines/session):
- Phase 0: 1 session
- Phase 1: 1 session
- Phase 2: 1 session
- Phase 3: 1 session
- Phase 4 + Phase 6: 1 session (parallel)
- Phase 5: 1 session
- Phase 7 + Phase 8: 1 session (parallel)
- **Total: ~7 sessions**

---

## Verification Gates

| Phase | Gate Test |
|-------|-----------|
| 0 | `apply_rules_fixpoint` terminates; `scalar_expand` distributes correctly |
| 1 | `R_{abcd} + R_{abdc} → 0` via canonicalize+collect; `R_{a[bcd]} = 0` via Bianchi rules |
| 2 | `[∇_a, ∇_b]V^c = R^c_{dab} V^d` derived, not hard-coded |
| 3 | `expand_perturbation(Perturbation(Ricci[-a,-b], 1))` matches hard-coded δRicci |
| 4 | Schwarzschild Riemann components match known values |
| 5 | Riemann→Weyl decomposition; Weyl tracelessness |
| 6 | `d(d(ω)) = 0`; Hodge★★ = ±1 |
| 8 | `Pkg.add("TensorGR")` works on clean machine |

---

## Reference Files

- xTensor: `reference/xAct/xAct/xTensor/xTensor.m` (10,468 lines)
- xPerm: `reference/xAct/xAct/xPerm/xPerm.m` (~1,750 lines)
- xPert: `reference/xAct/xAct/xPert/xPert.m` (476 lines)
- xCoba: `reference/xAct/xAct/xCoba/` (4,295 lines)
- xTras: `reference/xAct/xAct/xTras/` (5,165 lines)
- xTerior: `reference/xAct/xAct/xTerior/` (1,178 lines)
- Current TensorGR.jl: `src/` (2,362 lines), `test/` (2,150 lines)
