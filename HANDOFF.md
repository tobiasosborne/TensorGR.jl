# HANDOFF — 2026-03-20 (Session 7)

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

- **230 of 352 issues closed** (70 closed this session, mostly stale-issue reconciliation + new validation)
- **Full test suite: ALL PASS** (360,310+ tests, verified this session)
- All pushed to `master`, no uncommitted work
- `bd stats` for live counts, `bd ready` for available work
- **Note**: beads database was restored from stale backup (March 16 snapshot). ~80 issues from sessions 3-6 may still be open but have implementations. Close as encountered.

---

## What Was Done This Session (70 issues)

### New Validation Tests (9 test files, ~260 tests)
- **test_np_schwarzschild.jl** (49 tests): Kinnersley tetrad + all 12 spin coefficients + Weyl scalars. Ground truth: Teukolsky 1973 Eq 4.4-4.6.
- **test_np_kerr.jl** (53 tests): Full Kerr NP quantities (complex spin coefficients, a=0.5). Ground truth: Teukolsky 1973.
- **test_goldberg_sachs.jl** (40 tests): κ=σ=0 ⟺ Ψ₀=Ψ₁=0 at 5 points (Schwarzschild + Kerr).
- **test_dhost_horndeski.jl** (31 tests): DHOST class Ia reduces to Horndeski (degeneracy, DOF count, round-trip).
- **test_weyl_identity.jl** (1 test): C²=K-2Ric²+R²/3 string match.
- **test_riemann_identities.jl** (6 tests): Pair symmetry, first Bianchi, ∇G=0, Einstein trace, Kretschner.
- **test_eih_potential.jl** (35 tests): Full EFTofPNG Feynman pipeline — EIH 1PN coefficients 3/-7/-1/2 against Goldberger-Rothstein Eq 40.
- **test_invar_level4.jl** (11 tests): Derivative commutation for differential invariants.

### New Implementation
- **simplify_level4**: Invar Level 4 derivative commutation (src/invariants/simplify_levels.jl)

### Stale Issue Reconciliation (~55 closures)
- Closed ~55 issues that had implementations from prior sessions but were marked open due to stale beads backup
- Deferred 10 speculative core-touching issues (tetrad AST, FullSimplify, IndexFree, etc.)
- PSALTer epic auto-closed (all children done)

### Sign Convention Discovery
- Our (-,+,+,+) code negates l-type and compound spin coefficients from Teukolsky (+,-,-,-)
- n-type coefficients (ν,λ,μ,π) match Teukolsky because metric flip cancels the missing NP negative sign in our definitions
- Documented in test headers with full derivation

---

## Key Decisions / Lessons

### Carried from previous sessions
- **FullySymmetric(n)** takes slot numbers as varargs: `FullySymmetric(1,2,3,4)` NOT `FullySymmetric(4)`
- **make_rule** RETURNS rules but does NOT register them
- **NP tetrad rules**: use function-based RewriteRule
- **symmetrize** takes `Vector{Symbol}` not `Vector{TIndex}`
- **Bimetric massive mode**: Use `(1/(1+c²))(δg − δf)` NOT `(1/(1+c²))(c²δg − δf)`

### New this session
- **NP sign convention**: Spin coefficients in (-,+,+,+) are NOT simply negated from Teukolsky. l-type (κ,σ,ρ,τ) are negated; n-type (ν,λ,μ,π) match Teukolsky; compound (ε,γ,α,β) are negated. This is because our code defines all simple coefficients WITHOUT the standard NP negative sign on n-type.
- **Beads sync issues**: Different machines have divergent beads state. The backup/restore path loses closures made after the last backup. Always `bd backup` before switching machines.
- **Ground truth verification**: Downloaded Teukolsky 1973 and Goldberger-Rothstein 2006 to reference/papers/. All NP and EIH test values are string-matched against these papers.

---

## Ready Queue Highlights

```bash
bd ready -n 10   # see top priorities
```

**Remaining P2 tasks (genuine new implementation):**
- TGR-bgl.11: Abstract Poisson equation solver for PPN
- TGR-u19: BH-Pert2 radial source term assembly
- TGR-34t.4: Anisotropic perturbation decomposition on Bianchi I

**Deferred (speculative, no downstream need):**
- TGR-lej: Abstract tetrad indices in AST
- TGR-ulo.1: FullSimplification
- TGR-0o2: Spin coefficients as Ricci rotation coefficients
- TGR-2d4.2: Tetrad type with frame vectors
- TGR-xmm.2: IndexFree type

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
- NP Schwarzschild: Ψ₂ = -M/r³, ρ=+1/r (our convention), α=-β=+cotθ/(2√2r)
- EIH 1PN: L_EIH coefficients 3(v²), -7(v·v), -1/2(G²m(m₁+m₂)/r²) from Goldberger-Rothstein Eq 40

## Quick Commands

```bash
bd ready                    # see available work
bd stats                    # project health
julia --project -e 'using Pkg; Pkg.test()'  # full test suite
git log --oneline -10       # recent commits
```
