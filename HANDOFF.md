# HANDOFF — 2026-03-19 (Session 6)

## DO NOT DELETE THIS FILE. Read it completely before working.

## TOBIAS'S RULES — FOLLOW TO THE LETTER

1. **SKEPTICISM**: All subagent work, handoffs — verify everything twice.
2. **DEEP BUGS**: Deep, complex, interlocked. Do not underestimate.
3. **NO BANDAIDS**: Best-practices full solutions only.
4. **WORKFLOW**: 3 subagents before any core code change (xAct research + 2 solutions).
5. **REVIEW**: Rigorous reviewer agent after every core change. No exceptions.
6. **GROUND TRUTH**: Physics is ground truth, not pinned numbers. Tests may be suspect.
7. **TESTING**: Targeted only, or full suite in background.
8. **REPEAT RULES**: Repeat occasionally to maintain focus.
9. **DO NOT UNDERESTIMATE**: This is deeply nontrivial.

**Corollary**: Review xAct source (at `reference/xAct/`) BEFORE changing core modules.
**Max 2-3 subagents at a time.** Checkpoint regularly.
**USE MAX THINKING (opus) for all subagents.** Medium effort missed a bimetric sign bug this session.

---

## Current State

- **243 of 369 issues closed** (8 closed this session: 235 → 243)
- **Full test suite: ALL PASS** (verified this session)
- All pushed to `master`, no uncommitted work
- `bd stats` for live counts, `bd ready` for available work

---

## What Was Done This Session (8 issues)

### NP/GHP Formalism (3 issues)
- **TGR-900**: 18 NP field equations (Ricci identities, NP 1962 Eqs 4.2a–4.2r)
  - NPFieldEquation struct with symbolic RHS terms
  - Vacuum reduction, l↔n symmetry verified, Lambda in exactly 4 equations
  - 1128 tests
- **TGR-jvn**: 11 NP Bianchi identities (NP 1962 Eqs 4.5a–4.5k)
  - NPBianchiIdentity struct with Ricci derivative terms on LHS
  - 8 Weyl-sector + 3 contracted Bianchi identities
  - Cross-checked against PreludeAndFugue/newmanpenrose Python/SymPy
  - 962 tests
- **TGR-2gx**: 12 GHP field equations (GHP 1973 Eqs 3.3–3.12)
  - GHPFieldEquation struct with GHP weight tracking
  - Derived from NP eqs by absorbing improper coefficients (ε,γ,α,β)
  - Weight consistency verified for every term
  - 420 tests

### Bimetric Gravity (2 issues + 1 epic)
- **TGR-6s5**: Mass eigenstates diagonalization
  - bimetric_mass_eigenstates(bp) → (massless, massive, m2_FP)
  - bimetric_inverse_transform(bp) → (delta_g, delta_f)
  - **BUG CAUGHT**: Agent used wrong massive mode (c²δg−δf instead of δg−δf); round-trip failed for c≠1. Fixed during review.
  - 50 tests
- **TGR-bj3**: Higuchi bound validation (m² ≥ 2Λ/3)
  - higuchi_bound, higuchi_coefficient, is_higuchi_healthy
  - Partially massless point verified: coefficient = 0 at m² = 2Λ/3
  - 68 tests

### Clifford Algebra (1 issue + 1 epic auto-closed)
- **TGR-dai.7**: Gamma trace validation to order 6
  - Tr(γ^a...γ^f) recursive formula verified (15 terms at order 6)
  - γ⁵ traces, Fierz completeness, charge conjugation properties
  - 84 tests
- **TGR-dai epic auto-closed** (all children complete)

### Metric-Affine Gravity (1 issue)
- **TGR-swh.7**: Brauer algebra 11-piece irreducible decomposition
  - 6 symmetric + 5 antisymmetric pieces of MA Riemann tensor
  - Dimensions sum to 96 for d=4, completeness verified
  - 75 tests

---

## Key Decisions / Lessons

### Carried from previous sessions
- **FullySymmetric(n)** takes slot numbers as varargs: `FullySymmetric(1,2,3,4)` NOT `FullySymmetric(4)`
- **make_rule** RETURNS rules but does NOT register them. Use `register_rule!(reg, rule)` for function-based rules
- **NP tetrad rules**: use function-based RewriteRule (like soldering_form.jl), not pattern matching on products with dummy pairs
- **symmetrize** takes `Vector{Symbol}` not `Vector{TIndex}`
- **_is_zero name collision**: renamed to `_mp_is_zero` to avoid clash with petrov_classify.jl

### New this session
- **Bimetric massive mode**: Use `(1/(1+c²))(δg − δf)` NOT `(1/(1+c²))(c²δg − δf)`. The inverse δg = γ + c²χ, δf = γ − χ only round-trips correctly with the first convention.
- **Medium-effort subagents miss physics bugs**: A medium-effort agent noticed the round-trip failed for c≠1 and rationalized it as "expected". Always use max thinking (opus) for physics code.
- **NP Bianchi identities have 11 equations**, not 8: 8 Weyl-sector (4.5a–4.5h) + 3 contracted (4.5i–4.5k) involving only Ricci scalars.

---

## Ready Queue Highlights

```bash
bd ready -n 10   # see top priorities
```

**High-value P2 tasks:**
- TGR-v8u: EFTofPNG 1PN EIH potential (needs local paper research)
- TGR-ulo.2: SortCovDsToDiv (core pipeline — needs full workflow)
- TGR-xlu.5: DDI simplification pass (core pipeline — needs full workflow)
- TGR-lnp: Schwarzschild NP quantities validation (just unblocked)
- TGR-w0z: Kerr NP quantities validation (just unblocked)

**Newly unblocked by this session:**
- TGR-9t4: Goldberg-Sachs theorem test (needs NP field equations)
- TGR-lnp: Schwarzschild NP validation (needs NP field equations)
- TGR-w0z: Kerr NP validation (needs NP field equations)
- TGR-swh.8: Poincare gauge theory action (needs Brauer decomposition)

**Deferred:**
- TGR-avk: trace-free enforcement (deferred, see Session 5 notes)
- TGR-e04: _avoid removal (P4, deferred from earlier)

---

## Architecture Quick Reference

See CLAUDE.md for full details. Key points:

- **Core pipeline**: expand_products → contract_metrics → contract_curvature → canonicalize → collect_terms → apply_rules
- **DO NOT** sort TSum terms, sort deriv chains in canonicalize, or simplify bilinear products before kernel extraction
- **Registry**: thread-safe via `task_local_storage`, `with_registry(reg) do ... end`

## Physics Ground Truth

- K_FP: spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
- K_R²: spin2=0, spin0s=3k⁴, spin1=0, spin0w=0
- K_Ric²: spin2=1.25k⁴, spin0s=k⁴, spin1=0, spin0w=0
- Spin-1 and spin-0w MUST be zero for ALL kernels (diffeomorphism invariance)
- PPN scalar-tensor: gamma=(omega+1)/(omega+2), beta=1+Psi*omega'/(4(2omega+3)(omega+2)^2)
- Bimetric FP mass: m²_FP = m²(β₁+2cβ₂+c²β₃)/(1+c²)
- Higuchi bound: m² ≥ 2Λ/3 for massive spin-2 on dS

## Quick Commands

```bash
bd ready                    # see available work
bd stats                    # project health
julia --project -e 'using Pkg; Pkg.test()'  # full test suite
git log --oneline -10       # recent commits
```
