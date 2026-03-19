# HANDOFF — 2026-03-19 (Session 5)

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

---

## Current State

- **235 of 369 issues closed** (41 closed this session: 194 → 235)
- **Full test suite: ALL PASS** (verified this session)
- All pushed to `master`, no uncommitted work
- `bd stats` for live counts, `bd ready` for available work

---

## What Was Done This Session (41 issues)

### Spinor / NP / GHP Infrastructure (19 issues)
- Weyl spinor Psi_{ABCD}, Ricci spinor Phi_{ABA'B'}
- NP null tetrad with function-based contraction rules
- 5 Weyl scalars, 10 Ricci scalars, Lambda
- 12 NP spin coefficients (kappa through beta)
- Spin covariant derivative, irreducible decomposition
- Spinor Ricci identity (commutator), curvature decomposition
- GHP weights, derivative operators (thorn/edth), commutator relations
- NP directional derivatives, commutator table
- Spinor pipeline integration verified
- 5 duplicates closed (already implemented)

### PSALTer Particle Spectrum Chain (7 issues, epic completed)
- Moore-Penrose propagator via spin-sector decomposition
- Unitarity analysis (no-ghost, no-tachyon per sector)
- Source constraints from gauge invariance
- Maxwell validation: 1 massless spin-1, unitary, ∂_μJ^μ=0
- Proca validation: 3 massive spin-1 DOF, no ghost
- Fierz-Pauli unitarity: spin-2 healthy, spin-0s ghost (Boulware-Deser)
- Multi-field kernel extraction
- **TGR-4zw epic auto-closed** (all children complete)

### Physics Validations Against Local Papers
- PPN scalar-tensor: Hohmann (2021) arXiv:2012.14984 Eq (exppnpar)
  gamma=(omega+1)/(omega+2), beta=1+Psi*omega'/(4(2omega+3)(omega+2)^2)
- **TGR-bgl xPPN epic auto-closed** (all children complete)

### Metric-Affine Gravity Chain (3 issues)
- Non-metricity irreducible decomposition (Weyl vector, traces, traceless)
- Distortion tensor N = K + L (contortion + disformation)
- Metric-affine Riemann tensor (asymmetric Ricci, no pair symmetry)

### Clifford Algebra Chain (3 issues)
- Gamma^5 chirality (Gamma5 AST node, trace identities, slash notation)
- Fierz identities (5×5 rearrangement matrix, Nishi Table I)
- Charge conjugation matrix C

### Bimetric Gravity (1 issue)
- Linearized Hassan-Rosen: mass eigenstates, FP mass formula, rank-1 mass matrix

### Invar (1 issue)
- Level 3: Second Bianchi identity for differential invariants

### Deferred
- **TGR-avk** (trace-free enforcement): deferred. Already enforced via Weyl
  decomposition rules. Core pipeline change is high risk, low reward, no paper
  ground truth. Revisit only if a real computation fails.

---

## Key Decisions / Lessons

- **FullySymmetric(n)** takes slot numbers as varargs: `FullySymmetric(1,2,3,4)` NOT `FullySymmetric(4)`
- **make_rule** RETURNS rules but does NOT register them. Use `register_rule!(reg, rule)` for function-based rules
- **NP tetrad rules**: use function-based RewriteRule (like soldering_form.jl), not pattern matching on products with dummy pairs
- **symmetrize** takes `Vector{Symbol}` not `Vector{TIndex}`
- **_is_zero name collision**: renamed to `_mp_is_zero` to avoid clash with petrov_classify.jl

---

## Ready Queue Highlights

```bash
bd ready -n 10   # see top priorities
```

**High-value P2 tasks:**
- TGR-6s5: Bimetric mass eigenstates diagonalization (just unblocked by cp7)
- TGR-v8u: EFTofPNG 1PN EIH potential (needs local paper research)
- TGR-ulo.2: SortCovDsToDiv (core pipeline — needs full workflow)
- TGR-xlu.5: DDI simplification pass (core pipeline — needs full workflow)
- TGR-jvn: NP Bianchi identities in NP form
- TGR-900: NP 18 field equations (Ricci identities)

**Deferred:**
- TGR-avk: trace-free enforcement (deferred, see above)
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

## Quick Commands

```bash
bd ready                    # see available work
bd stats                    # project health
julia --project -e 'using Pkg; Pkg.test()'  # full test suite
git log --oneline -10       # recent commits
```
