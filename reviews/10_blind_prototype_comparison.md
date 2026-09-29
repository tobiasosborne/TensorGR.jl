# Blind Prototype Comparison — IR Canonicalizer in C (Sonnet vs Opus)

**Date**: 2026-09-29 · **Author**: Claude (Opus 5.5), comparing two blind subagent implementations
**Brief**: identical C brief (IR canonicalizer for tensor monomials, SeQuant-style graph encoding, see
`08_canonicalization_literature_survey.md` §10 and `09_canonicalization_perf_bound.md`)

| | Sonnet | Opus |
|---|---|---|
| Branch (local, not pushed) | `proto/canon-ir-baseline` @ `302bd12` | `proto/canon-ir-baseline-opus` @ `a5351dc` |
| Code | `proto/canonir/` | `proto/canonir/` |
| Write-up | `proto/canonir/RESULTS.md` (519 lines) | `proto/canonir/RESULTS.md` |
| Core source | 596 lines C + 70-line header | 968 lines C + 121-line header |
| Tests / bench tooling | 423 / 174 lines | 681 / 522 lines |

Both implement classic nauty-style individualization–refinement (not Jefferson–Waldecker–Wilson graph
backtracking): equitable refinement, first smallest non-singleton target cell, lexicographically smallest
leaf certificate, automorphism/orbit pruning, signs from per-type signed slot groups plus Grassmann
parity, zero iff an odd automorphism exists. Neither implements the antisymmetric-metric (spinor ε)
stretch goal, multi-term (Bianchi) symmetries, or certificate output.

---

## 1. Verification performed by the main session

**Own test suites** (clean rebuild, run by me, not by the agents):

| | Sonnet | Opus |
|---|---|---|
| `make test`, `-O2` | 1,016,096 checks, 0 failures | 319,035 checks, 0 failures |
| `make test`, ASan + UBSan | 1,016,096 checks, 0 failures | 319,035 checks, 0 failures |

Each suite checks against its own brute-force oracle (n ≤ 12), invariance under random group moves, and
regression cases from review 07. Sonnet also ran a 7-mutant mutation test (5 killed strongly, 1 weakly
— the best-leaf zero-check path — and 1 equivalent mutant).

**Blind cross-check** (`10_probes/xcheck.c`): both libraries linked into one program and run on
identical random monomials (14 tensor types incl. Riemann, (anti)symmetric blocks, mixed blocks,
Grassmann-odd vectors/forms; palettes forcing identical factors). Checks:
- zero detection agrees;
- for pairs related by random symmetry moves (factor reordering, slot-symmetry moves, dummy renaming,
  up/down swaps): both identify them as equivalent, and relative signs agree;
- for pairs with randomly re-paired dummies (mostly inequivalent): both agree on equivalence and
  relative sign;
- **absolute sign cross-feed**: feeding each implementation's canonical output into the other must
  reproduce the other's canonical form with composed signs `sO(canonS(x)) · sO(x) = sS(x)`, and vice versa.

| Monomials | Max slots | Zeros | Inequivalent repair pairs | Checks | Disagreements |
|---|---|---|---|---|---|
| 100,000 | 12 | 34,323 | 27,695 | 1,009,539 | **0** |
| 100,000 | 20 | 36,812 | 34,171 | 974,559 | **0** |
| 30,000 | 32 | 11,143 | 11,562 | 289,554 | **0** |
| 5,000 | 64 | 1,937 | 2,012 | 47,238 | **0** |
| 1,000 | 128 | 398 | 422 | 9,330 | **0** |

Total ≈ 2.33 M checks, 0 disagreements, also clean under ASan + UBSan on a 7,000-monomial run.
**Harness sensitivity**: flipping 1 in 500 of one implementation's signs produced 9 detected failures
on a 3,000-monomial run. Two independently written implementations agreeing on this volume — including
instances up to 128 slots, far beyond either oracle's reach — is strong evidence both are correct for the
supported symmetry classes. It is not a proof: both share the same algorithm family and could share a
conceptual blind spot (e.g. symmetry types outside the brief).

---

## 2. Head-to-head performance (identical inputs, same session, interleaved)

Pinned to cpu0 (P-core), AC power, shared lock, 31 interleaved batches alternating order, median.
Effective clock from the L1 pointer chase: 3.04–3.13 ns for 5 cycles ⇒ ≈ 1.6 GHz.

| Workload | Sonnet (ns/call) | Opus (ns/call) | Sonnet / Opus |
|---|---|---|---|
| random, ≤ 12 slots (1024 monomials) | 4,808 | 4,119 | 1.17 |
| random, ≤ 24 slots | 10,678 | 8,795 | 1.21 |
| random, ≤ 48 slots | 26,199 | 19,381 | 1.35 |
| Riemann chain k = 3 (64 relabellings) | 15,126 | 7,936 | 1.91 |
| Riemann chain k = 8 | 73,278 | 36,642 | 2.00 |
| Riemann chain k = 12 | 167,178 | 66,182 | 2.53 |
| $(R_{abcd}R^{abcd})^2$ | 30,687 | 15,976 | 1.92 |
| $(R_{abcd}R^{abcd})^4$ | 132,419 | 57,530 | 2.30 |
| $(R_{abcd}R^{abcd})^6$ | 376,265 | 116,510 | 3.23 |

**Opus is faster everywhere: 1.2–1.35× on random terms, 1.9–3.2× on highly symmetric products**, and
the gap grows with symmetry. The search statistics explain it: Opus detects most automorphisms
*implicitly* (without descending to a leaf) and prunes with stored automorphisms at every node, so it
visits 3 leaves on every Riemann chain versus k + 3 for Sonnet (e.g. k = 8: 38 nodes / 3 leaves vs
55 nodes / 11 leaves). Opus found the need for every-node pruning itself: its invariance tests exposed
a factorial blow-up in its first design.

**Against the speed-of-light model** (each agent's own locked sessions, cycles):
Opus 5–15× above the practical floor, Sonnet 6–18×. **Against the old TensorGR path** (Julia + xperm,
known to be wrong on some inputs): ≈ 35–50× faster on Riem³, ≈ 60–145× on Riem⁴ — a system-level
comparison, not an algorithmic one.

---

## 3. Findings that feed back into the model (`09`)

- **The model's search term assumed the identical factors are permuted by S_k. That is wrong for real
  instances.** In a closed Riemann chain the factor permutations are only rotations and reflections
  (dihedral), but there are extra per-link symmetries (both agents report automorphism counts ≈ k + 2).
  Measured chain scaling: n^1.3–1.4 (Opus), n^1.7 (Sonnet) vs the model's n^1.8. For freely
  exchangeable pairs, $(R_{abcd}R^{abcd})^m$, Opus measures n^1.76–1.91, matching the model's ≈ n².
  §4 of `09` should parametrize the search cost by the actual automorphism group, not k.
- **Instruction counts are 19–35× the model's µop counts** (Sonnet, valgrind). The model's per-primitive
  costs are far below what a straightforward implementation executes — the gap to the floor is mostly
  constant-factor overhead (graph build, sorting, certificate handling), not search. Opus: refinement
  ≈ 55% of Riem⁸, fixed costs ≈ 50% of the distinct-factor case.
- **Sonnet's optimization pass cut instructions 5–12% with no measurable wall-clock gain; Opus's cut
  cycles 1.3–1.9×.**
- **The laptop is a poor benchmark host**: f_eff drifted 0.7–2.3 GHz within sessions. Relative
  comparisons in the same interleaved session (§2) are trustworthy; absolute ns are ±30%.

---

## 4. Assessment

| Criterion | Sonnet | Opus |
|---|---|---|
| Correctness (own tests + cross-check) | ✓ | ✓ |
| Test depth | more checks (1.0 M), mutation testing | fewer checks (0.32 M), larger instances (to n = 128), found and fixed a real pruning bug |
| Speed, random terms | baseline | 1.2–1.35× faster |
| Speed, symmetric products | baseline | 1.9–3.2× faster |
| Pruning | orbit pruning, leaf-based automorphisms | implicit automorphisms + stored-automorphism pruning at every node |
| Code size | smaller (596 lines core) | larger (968 lines core) |
| Limits | 128 slots, 64 factors, rank ≤ 8 | 128 slots, 128 factors, rank ≤ 32 |

**Recommendation**: take the Opus implementation as the base, and port Sonnet's mutation-testing
script and higher-volume random checks onto it. Keep `10_probes/xcheck.c` as a permanent differential
test (two independent implementations) while the core evolves.

**Next steps** (not started):
1. Bitset refinement for ≤ 128 slots and a discrete-refinement fast path (both agents list these).
2. Antisymmetric metric (spinor ε) signs; multiple vector bundles; derivative slots (∂ commuting,
   ∇ not).
3. Certificate output compatible with a `hex-graph-iso`-verified configuration (survey §9).
4. Julia binding via `ccall`, then golden-master comparison against xAct (`test/golden/`).
5. Re-run the benchmarks on a desktop with a fixed governor; revise `09` §4 search model.

**Reproduce**: `bash reviews/10_probes/build_and_run.sh` from the repo root (needs both local branches).

---

## 5. Addendum — one hard instance vs xperm.c (quick and dirty)

**Instance**: $(R_{abcd}R^{abcd})^m$ — 2m identical, freely exchangeable Riemann tensors, all indices
contracted: the identical-factor case Niehoff flags as factorial for Butler–Portugal. xperm.c is called
directly from C (`canonical_perm`, full double-coset mode with dummies and symmetric metric — *not* the
all-free mode TensorGR uses), with slot group generated by each Riemann's (12)⁻, (34)⁻, (13)(24) plus
adjacent identical-block swaps. Convention (verified empirically on $R_{abcd}R^{abcd}$ vs $R_{abcd}R^{bacd}$):
`PERM[name] = slot`, dummy pairs given as the slots of (up, down). Correctness gate: all three solvers give
one canonical form (and consistent relative signs) across 64 random relabellings of the instance
(xperm gated at m = 1, 2; at m = 3 only its first call's sign is compared). Harness: `10_probes/hard.c`.
Single P-core (cpu0), AC power, effective clock ≈ 1.4–2.0 GHz during the runs.

| m | factors / slots | Sonnet | Opus | xperm, SGS cached | xperm, stock (Schreier–Sims per call) | Opus speed-up vs xperm (cached) |
|---|---|---|---|---|---|---|
| 1 | 2 / 8 | 10.5 µs | 5.6 µs | 98 µs | 206 µs | ≈ 18× |
| 2 | 4 / 16 | 32.2 µs | 17.6 µs | 3.75 ms | 7.98 ms | ≈ 210× |
| 3 | 6 / 24 | 60.4 µs | 28.6 µs | **5.48 s** (one call) | 5.48 s | **≈ 190,000×** |

xperm grows factorially with the number of identical factors (×38 from m = 1→2, ×1,460 from 2→3),
while both IR prototypes grow roughly linearly here. Caveats: one instance family, one run per point,
default xperm base ordering (a smarter base or Niehoff's modifications would help xperm, but vendored
xperm has neither), and xperm allocates per call. Raw output: `10_probes/hard_m1_m2.txt`, `10_probes/hard_m3.txt`.
