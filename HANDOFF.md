# HANDOFF — 2026-03-19

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

- **182 of 362 issues closed** (51 closed last session)
- **356,800+ tests, 0 failures**
- All pushed to `master`, repo cleaned (no stale branches/files)
- `bd stats` for live counts, `bd ready` for available work

---

## ACTIVE SIDE QUEST: xAct Ground Truth Test Suite

**Priority**: Finish this before returning to feature work.

### What It Is

xAct (the Mathematica CAS for tensor algebra) has NO official test suite. We are building one using published physics papers that used xAct. Each test reproduces a specific equation from a paper using TensorGR.jl and string-matches the output.

### Protocol (Tobias's orders)

1. Deep-research all xAct API features and document them
2. Find canonical published physics papers that used each feature
3. Papers MUST be downloaded locally (all are in `reference/papers/`)
4. Equations that xAct produced MUST be extracted from LOCAL TeX sources — no hallucination
5. A TensorGR.jl program MUST compute the same thing — NO PRINTLN CHEATING
6. The test passes if string match occurs between TensorGR output and paper equation

### What's Done

**12 papers downloaded** (PDF + TeX source) to `reference/papers/`:

| Paper | arXiv | xAct Feature | TeX Source |
|-------|-------|-------------|------------|
| Nutma 2014 | 1308.3493 | xTras (contractions, invariants, Euler density) | `1308_3493_src/xTras.tex` |
| Brizuela 2009 | 0807.0824 | xPert (perturbation formulas) | `brizuela_src/xPert.tex` |
| Barker 2024 | 2406.09500 | PSALTer (spin projections) | already local |
| Bueno Cano 2016 | 1607.06463 | Higher-derivative gravity | `buenocano_src/` |
| Buoninfante 2020 | 2012.11829 | Higher-derivative gravity | `buoninfante_src/` |
| Hohmann 2021 | 2012.14984 | xPPN (PPN parameters) | `2012_14984_src/xppnpaper.tex` |
| Pitrou 2013 | 1302.6174 | xPand (cosmological perturbations) | `1302_6174_src/xPand_-_arXiv2.tex` |
| Garcia-Parrado 2012 | 1110.2662 | Spinors package | `1110_2662_src/SpinorsCPC.tex` |
| Levi & Steinhoff 2017 | 1705.06309 | EFTofPNG | `1705_06309_src/revised3.tex` |
| Tattersall 2018 | 1711.01992 | BH perturbations in modified gravity | `1711_01992_src/final_jan18.tex` |
| Agullo 2020 | 2006.03397 | Bianchi I perturbations | `2006_03397_src/main.tex` |
| Casalino 2020 | 2003.07068 | Regularized Lovelock | `2003_07068_src/main.tex` |

**Equations extracted** from 2 papers so far:

- **Nutma (xTras)**: 13 computable equations cataloged. Key: Gauss-Bonnet E₄, Weyl decomposition, variational δR/δg, AllContractions counts (1 Riemann → 1 scalar, 2 Riemanns → 4 scalars), Euler densities d=2,4,6,8, linearized Einstein (10 terms).
- **Brizuela (xPert)**: 14 equations cataloged. Key: inverse metric perturbation (Eq 6), Christoffel perturbation (Eq 7), Riemann perturbation (Eq 9/10), Ricci perturbation (Eq 11), Ricci scalar perturbation (Eq 12), linearized Einstein tensor (10 terms).

### What's Next (for you to do)

1. **Extract equations from remaining 10 papers** — same protocol as Nutma/Brizuela. Read the TeX source, find equations computed by xAct, record equation number + LaTeX.

2. **Write test file `test/test_xact_ground_truth.jl`** with sections per paper:
   - Each test: set up registry, compute expression with TensorGR, compare output
   - Use `to_latex()` or structural comparison, NOT println
   - Group by xAct subpackage (xTensor, xPert, xCoba, xTras, Invar, Spinors, etc.)

3. **Priority equations to implement first** (these TensorGR CAN compute today):
   - Gauss-Bonnet: `euler_density(:g)` == R² - 4R_{ab}R^{ab} + R_{abcd}R^{abcd}
   - Weyl decomposition: `riemann_to_weyl` matches Nutma formula
   - δR/δg^{ab}: `variational_derivative` matches Nutma Eq (line 1512)
   - δ¹g^{ab} = -h^{ab}: first-order inverse metric perturbation
   - δ¹Ric_{ab}: first-order Ricci perturbation
   - δ¹R: first-order Ricci scalar perturbation
   - PPN: γ_GR=1, γ_BD=(1+ω)/(2+ω) (already tested but need paper citation)
   - Spin projector completeness: P2+P1+P0s+P0w = identity

4. **Beads issues created**: TGR-4y6 (Nutma), TGR-1rq (Brizuela), TGR-6b8 (Hohmann), TGR-rzi8 (Barker), TGR-gmot (Pitrou)

### Key Gaps (TensorGR vs xAct — equations we CANNOT yet reproduce)

- **ConstructDDIs**: dimensional dependent identity enumeration (not implemented)
- **FullSimplification**: Invar-level multi-term Bianchi scalar monomial reduction
- **delta_einstein**: no dedicated function (compose from delta_ricci + delta_ricci_scalar)
- **Multi-order h^{(n)}**: TensorGR locked to background field method (single h)
- **Higher-d Euler densities**: d≥6 not verified (d=4 works)
- **RiemannYoungProject**: automated Riemann Young tableau projection
- **Determinant perturbation**: δⁿ(det g) not implemented

---

## Architecture Quick Reference

See CLAUDE.md for full details. Key points:

- **Core pipeline**: expand_products → contract_metrics → contract_curvature → canonicalize → collect_terms → apply_rules
- **xperm.c FFI**: Butler-Portugal canonicalization via C library at `deps/libxperm.so`
- **Registry**: thread-safe via `task_local_storage`, `with_registry(reg) do ... end`
- **DO NOT** sort TSum terms, sort deriv chains in canonicalize, or simplify bilinear products before kernel extraction

## Physics Ground Truth

- K_FP: spin2=2.5k², spin0s=-k², spin1=0, spin0w=0
- K_R²: spin2=0, spin0s=3k⁴, spin1=0, spin0w=0
- K_Ric²: spin2=1.25k⁴, spin0s=k⁴, spin1=0, spin0w=0
- Spin-1 and spin-0w MUST be zero for ALL kernels (diffeomorphism invariance)
- Horndeski DHOST: a1 = G4_X, a2 = -G4_X, a3=a4=a5=0
- RInv canonicalization is CONJUGATION, not left-action
- _avoid in perturbation engine is correct (deferred to P4, not a bug)

## Quick Commands

```bash
bd ready                    # see available work
bd stats                    # project health
bd show <id>                # issue details
julia --project -e 'using Pkg; Pkg.test()'  # full test suite (~7min)
git log --oneline -10       # recent commits
```

## Session Close Protocol

```
[ ] git status
[ ] git add <files>
[ ] git commit -m "..."
[ ] git push
[ ] bd close <completed issues>
```
