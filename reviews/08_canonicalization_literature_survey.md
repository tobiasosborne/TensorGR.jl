# Canonicalization Core — Literature Survey for an xperm.c Replacement

**Author**: Claude (Opus 5.5)
**Date**: 2026-09-29
**Purpose**: identify the most up-to-date algorithms to implement in a new, optimised canonicalization core
that replaces `deps/xperm.c` and goes beyond it: tensor index canonicalization (slot symmetries,
dummies, metric and spinor-metric signs, Grassmann signs), multi-term symmetries, Feynman diagrams,
Feynman-integral topologies, fermionic operator networks, gamma matrices — with formal-methods backing.

**Conventions**: every paper/tool cited below was retrieved during this survey (link given).
Statements marked **[synthesis]** are my own analysis, not claims from a source.
Context: `07_julia_best_practices_review.md` §1 showed the current core misuses xperm (all indices
passed as free), producing wrong results.

---

## 0. Executive summary

1. **The field has split into two families, and the newer evidence favours the graph family for
   the workloads that matter.** The *group-theoretic* family (Butler–Portugal double-coset
   canonicalization; xPerm, SymPy, Cadabra, Niehoff's improvements) represents symmetries as
   permutation groups given by generators. The *graph-theoretic* family encodes a tensor monomial or
   network as a coloured graph and calls an individualization–refinement (IR) canonical labeller
   (nauty/Traces/bliss-style). SeQuant (Nov 2025) measured its graph canonicalizer at ≈ n² on networks
   of identical tensors where two Butler–Portugal implementations blew up — precisely the case
   Niehoff (2018) identified as a remaining factorial blind spot of Butler–Portugal (products of
   identical factors `T_abc T_def T_ghi …`). Products of identical Riemann tensors are the bread and
   butter of GR invariants.
2. **The modern theory unifies both families.** *Graph backtracking* (Jefferson–Waldecker–Wilson
   2023, implemented in Rust as Vole) computes canonical images under an arbitrary permutation group
   and explicitly generalises both nauty and Linton's minimal-image algorithm. Schweitzer–Wiebking
   (STOC 2019) give a general canonization framework for combinatorial objects (graphs, hypergraphs,
   codes, permutation groups, …). A single engine can therefore serve tensors, diagrams, and
   polynomials.
3. **Everything on your list reduces to "canonical form of a coloured structure under a group,
   plus a sign character"** **[synthesis]**: tensor monomials (slot group × dummy relabelling,
   signs from antisymmetry / spinor metric / Grassmann parity), Feynman diagrams (coloured
   multigraph; |Aut| = symmetry factor; fermion-loop and external-leg signs), Feynman-integral
   families (Pak's polynomial canonicalization = canonical form of a coefficient/exponent matrix
   under column permutations), fermionic tensor networks after Wick contraction (SeQuant). Gamma
   matrices are the exception: they need an algebraic normal form first (antisymmetrised basis
   Γ^{a₁…a_k}), after which the result is an ordinary monoterm-symmetric object.
4. **Multi-term symmetries (Bianchi, cyclic, dimension-dependent identities) are not a
   canonical-labelling problem.** They require either Young-projector methods (Cadabra, Alakazam),
   graph-algebra extension (Li–Li–Li 2017), group averaging (Kryukov–Shpiz), or a database of
   linear relations (xAct Invar). Treat them as a linear-algebra layer *on top of* monoterm
   canonicalization **[synthesis]**.
5. **Certification is feasible today.** Banković–Drecun–Marić (LMCS 2023) turned the McKay–Piperno
   canonical-labelling algorithm into a proof system with exportable proofs, independently
   checked, with soundness formalised in Isabelle/HOL. That is the template for a
   Lean-checkable core: *emit certificates, verify the checker*, rather than verify the search.
   Better still, `leanprover/hex-graph-iso` (Sept 2026) already provides **verified Lean 4
   implementations of pinned nauty 2.9.3 configurations** for coloured graphs, with a kernel-checked
   `graph_iso` tactic (§9) — pinning the new core's canonical function to one of those configurations
   would give Lean-side canonicity essentially for free.
6. **Competitive landscape in 2025–26 is active, including in Julia.** Alakazam.jl (Aug 2026,
   graph-isomorphism based, grading, gamma matrices, Young projection) and GraphCombinations.jl
   (allocation-free Julia canonical labeller for coloured directed multigraphs) exist; Symbolica
   (Rust) canonizes tensors via its MIT-licensed `graphica` crate; FeynGraph (Rust, 2025) generates
   diagrams ~10–100× faster than QGRAF with correct multi-fermion signs.

**Recommendation in one line** **[synthesis]**: build one IR/graph-backtracking canonical labeller
over coloured relational structures with signed-group gadgets and certificate output; put domain
encoders (tensor monomial, diagram, polynomial, TN) in front of it; keep Butler–Portugal-style and
Niehoff-style fast paths only where profiling proves them faster; handle multi-term identities in
a separate linear-algebra layer. Details and open questions in §10–11.

---

## 1. Problem statement, unified **[synthesis]**

A canonicalizer is a function `canon(x)` on objects `x` acted on by a group `G` such that
`canon(x) = canon(y) ⟺ y ∈ G·x`, together with a witness `g` with `g·x = canon(x)`. Our
instances:

| Domain | Object `x` | Group `G` | Sign character | Extra output |
|---|---|---|---|---|
| Tensor monomial | slot → index assignment | slot symmetries `S` (per tensor, plus factor permutations of identical/commuting factors) × dummy relabelling `D` (renaming, and swapping up/down within a pair when a metric exists) | antisymmetry, spinor metric ε (up/down swap = −1), Grassmann parity of factors | "zero by symmetry" when an automorphism has sign −1 |
| Feynman diagram | coloured multigraph with external legs | vertex relabelling | fermion loops, external fermion ordering | `|Aut|` → symmetry factor |
| Integral family | Symanzik `U`, `F` polynomials (coeff./exponent matrix) | permutation of Feynman parameters (+ momentum shifts upstream) | none | mapping between families |
| Fermionic TN (post-Wick) | tensors + operator strings | tensor/slot/dummy permutations | fermionic reordering | phase |

The double-coset formulation of Butler–Portugal (`S·g·D`) is the special case where `G` is given by
generators. The graph formulation encodes `S` into gadgets and lets `Aut` of the coloured graph
realise both `S` and `D` implicitly. The **sign** is the non-standard ingredient: it is a character
`χ: Aut(x) → {±1}`; if `χ` is non-trivial on `Aut(x)` the object vanishes. Graph canonical labellers
compute generators of `Aut(x)`, so this test costs one evaluation per generator.

---

## 2. Group-theoretic family (the xperm lineage)

| Work | Contribution | Link |
|---|---|---|
| Butler, *Fundamental Algorithms for Permutation Groups* (1991) | double-coset canonical representatives | cited in SeQuant ref. 75 |
| Portugal, "An algorithm to simplify tensor expressions" (1998) | first use for tensors | [gr-qc/9803023](https://arxiv.org/pdf/gr-qc/9803023) |
| Manssur, Portugal, Svaiter, "Group-theoretic approach … II. Dummy indices" | dummy-index relabelling group | [math-ph/0107032](https://arxiv.org/html/math-ph/0107032) |
| Manssur & Portugal, *Canon* package (CPC 2004) | fast kernel | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010465503004946) |
| Martín-García, "xPerm: fast index canonicalization" (CPC 2008) | the C code we vendor; signed perms on n+2 points, metric symmetric/antisymmetric/none dummies | [arXiv:0803.0862](https://arxiv.org/pdf/0803.0862) |
| SymPy `tensor_can` | Python port; commuting/anticommuting tensors, symmetric/antisymmetric metrics | [docs](https://docs.sympy.org/latest/modules/combinatorics/tensor_can.html) |
| Niehoff, "Faster tensor canonicalization" (CPC 228, 2018) | polynomial for totally (anti)symmetric slot subsets; O(1)/O(n) label-group data structures; faster average case | [arXiv:1702.08114](https://arxiv.org/abs/1702.08114) |
| Diehl, `butler-portugal` Rust crate (v0.3) | modern Rust BP on signed perm groups, Schreier–Sims chains | [GitHub](https://github.com/sdiehl/butler-portugal), [blog](https://www.stephendiehl.com/posts/tensor_canonicalization_rust/) |

**Niehoff's own statement of the limits** (§4.4 of the paper): the factorial blow-up remains for
slot groups generated by *block* exchanges, e.g. `S = ⟨(1 3)(2 4), (3 5)(4 6), …⟩`, and he names
products of identical factors `T_abc T_def T_ghi ⋯` as a case needing special handling; detecting
length-k block symmetries in general "requires O(n^{k+1}) group membership tests … removing one head
from the hydra only springs forth more." This is the structural argument for switching families.

---

## 3. Graph-theoretic family

| Work | Contribution | Link |
|---|---|---|
| Obeid, *On the Simplification of Tensor Expressions*, MSc thesis, Waterloo (2001) | earliest graph approach | [PDF](https://cs.uwaterloo.ca/~smwatt/home/students/theses/NObeid2001-msc.pdf) |
| Bolotin & Poslavsky, *Redberry* (2013) | "index isomorphism" strategy; per Niehoff, the only package using it at the time | [arXiv:1302.1219](https://arxiv.org/pdf/1302.1219) |
| Li, Li, Li, "Riemann tensor polynomial canonicalization by graph algebra extension" (ISSAC 2017) | graph algebra + extension theory to handle **multi-term** symmetries of Riemann polynomials | [arXiv:1701.08487](https://arxiv.org/pdf/1701.08487), [ACM](https://dl.acm.org/doi/10.1145/3087604.3087625) |
| Kryukov & Shpiz, "Simplification of tensor expressions in computer algebra" (2018) and "The method of coloured graphs…" (Programming & Computer Software, 2021) | canonical form by averaging over a "signature stabilizer"; coloured graphs with typed (up/down) indices; linear relations; efficient on Riemann expressions | [arXiv:1811.07701](https://arxiv.org/abs/1811.07701), [Springer](https://link.springer.com/article/10.1134/S0361768821010102) |
| Gaudel, …, Köhn, Valeev, *SeQuant* I (Nov 2025, rev. Feb 2026) | graph TN canonicalizer on **bliss**: vertices for tensor cores, indices, slots, and slot *bundles* (bra/ket/column/proto-index); colours encode identity; symmetric bundles share colours; hyperedges and index dependencies; phase accumulated for antisymmetric bundles; dummies regenerated in order of appearance; ≈ n² on identical-tensor networks where BP variants explode; optional cosmetic lexicographic pass | [arXiv:2511.09943](https://arxiv.org/abs/2511.09943) |
| Ruijl, *Symbolica* (1.0, Nov 2025) + `graphica` crate (MIT) | tensor canonization via graph canonization, incl. cyclic symmetry and nested tensors; modified McKay algorithm; mixed directed/undirected multigraphs with arbitrary node/edge data; |Aut| and orbits | [release post](https://symbolica.io/posts/stable_release/), [graphica](https://github.com/symbolica-dev/graphica), [FORM 2025 slides](https://conference.ippp.dur.ac.uk/event/1459/contributions/8140/attachments/6451/8763/form2025.pdf) |
| Woods, *Alakazam* (Julia, Aug 2026) | "graph isomorphism over the Butler–Portugal algorithm" for `hard_simplify`; linear-time signature hashing for term collection; grading with inversion-count signs; Young projection for multi-term (`multiterm=true`, exponential); gamma-matrix contraction; claims orders-of-magnitude speed-ups vs Cadabra, xAct, Redberry | [arXiv:2608.20452](https://arxiv.org/abs/2608.20452), [GitLab](https://gitlab.com/B0bGary/alakazam.jl) |
| Zucker, "Tensors and graphs: canonization by search" (blog, Nov 2024) | accessible exposition: variable canonization under AC operations ≈ graph canonization | [post](https://www.philipzucker.com/canon_search/) |

**Why graphs win on identical factors** **[synthesis]**: in BP, exchanging whole identical factors is
a block permutation of slots that must be enumerated through the group; in the graph encoding,
identical tensors are isomorphic coloured subgraphs, and colour refinement plus orbit pruning
discovers the exchange as an automorphism without enumerating it.

---

## 4. Graph canonical-labelling engines

| Engine | Notes | Link |
|---|---|---|
| nauty / Traces (McKay & Piperno) | reference IR implementations; Traces strongest on hard coloured families | [Practical Graph Isomorphism II, arXiv:1301.1493](https://arxiv.org/abs/1301.1493); [site](https://pallini.di.uniroma1.it/) |
| bliss (Junttila & Kaski) | large sparse graphs; used by SeQuant | [docs](https://users.aalto.fi/~tjunttil/bliss/definitions.html) |
| dejavu (Anders & Schweitzer; 2.0, MIT) | state-of-the-art **automorphism** computation (random Schreier, probabilistic); the 2.0 documentation does not advertise canonical labelling | [docs](https://www.automorphisms.org/documentation/), [probabilistic test arXiv:2011.09375](https://arxiv.org/pdf/2011.09375), [parallel, ESA 2021](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2021.6) |
| sassy (Anders, Schweitzer, Stieß, SEA 2023) | preprocessor shrinking sparse substructures before symmetry detection | [arXiv:2302.06351](https://arxiv.org/abs/2302.06351) |
| graphica (Rust, MIT) | multigraphs, mixed directed edges, node/edge data, generation | [GitHub](https://github.com/symbolica-dev/graphica) |
| GraphCombinations.jl (Julia, v0.4.0) | exact coloured directed-multigraph canonical labelling; loops, multiplicities, fixed vertices, disconnected graphs; allocation-free warmed workspaces; levelwise and recursive-stabilizer kernels; built for physics diagram workloads (KeldyshContraction.jl migrated off nauty, reported ≈0.75× time on 14,347 cases) | [PR #208](https://github.com/oameye/GraphCombinations.jl/pull/208), [issue #157](https://github.com/oameye/GraphCombinations.jl/issues/157), [KeldyshContraction #411](https://github.com/oameye/KeldyshContraction.jl/issues/411) |

Complexity: IR is exponential in the worst case (Neuen & Schweitzer lower bound, SeQuant ref. 121),
polynomial for bounded-valence graphs, and Babai's quasipolynomial canonical form (SeQuant ref. 106)
exists in theory but is not practical. For tensor monomials (valence ≤ slot count, ≤ ~100 vertices)
IR is effectively polynomial in practice **[synthesis, consistent with SeQuant's measurement]**.

---

## 5. Canonical images in permutation groups (the unifying theory)

| Work | Contribution | Link |
|---|---|---|
| Jefferson, Jonauskytė, Pfeiffer, Waldecker, "Minimal and canonical images" (J. Algebra 2019) | canonical image of a set under a permutation group using orbit structure and subgroup chains; correctness proofs; GAP `images` package | [arXiv:1703.00197](https://arxiv.org/abs/1703.00197), [images](https://gap-packages.github.io/images/) |
| "Permutation group algorithms based on directed graphs" (2021; PEAL group) | graph backtracking: partition backtrack extended with graphs | [arXiv:2106.13132](https://arxiv.org/pdf/2106.13132) |
| "Perfect refiners for permutation group backtracking algorithms" | refiners that prune perfectly for common problems | [arXiv:2112.05065](https://arxiv.org/pdf/2112.05065) |
| Jefferson, Waldecker, Wilson, "Computing canonical images in permutation groups with graph backtracking" (2023) | canonical images under **arbitrary** groups; generalises nauty and Linton's minimal-image algorithm | [arXiv:2209.02534](https://arxiv.org/abs/2209.02534) |
| Vole (GAP package written in Rust, v0.6.0, Jan 2025) | high-performance graph backtracking: canonical images, stabilisers, intersections | [Vole](https://peal.github.io/vole/) |
| Schweitzer & Wiebking, "A unifying method for the design of algorithms canonizing combinatorial objects" (STOC 2019) | canonization for hereditarily-finite-set objects (graphs, hypergraphs, relational structures, codes, permutation groups) | [arXiv:1806.07466](https://arxiv.org/abs/1806.07466) |

**Why this matters for the core** **[synthesis]**: the tensor problem is "canonical image of a
labelled object under a group given partly by generators (slot symmetries) and partly implicitly
(dummy/factor exchange)". Pure nauty needs the generator part encoded as graph gadgets; pure BP
needs the implicit part enumerated. Graph backtracking accepts both — generators *and* graph
constraints — in one search, making it the most general candidate for the engine's search layer.
Vole is the reference implementation to benchmark against.

---

## 6. Multi-term symmetries (Bianchi, cyclic, dimension-dependent)

| Approach | Where | Link |
|---|---|---|
| Young projectors make multi-term symmetries manifest | Cadabra (Peeters) | [cs/0608005](https://arxiv.org/pdf/cs/0608005), [hep-th/0701238](https://arxiv.org/pdf/hep-th/0701238) |
| Young projection, opt-in, exponential cost | Alakazam | [arXiv:2608.20452](https://arxiv.org/abs/2608.20452) |
| Graph algebra extension for Riemann polynomials | Li–Li–Li | [arXiv:1701.08487](https://arxiv.org/pdf/1701.08487) |
| Averaging over signature stabilizer + linear relations | Kryukov–Shpiz | [arXiv:1811.07701](https://arxiv.org/abs/1811.07701) |
| Precomputed relation databases for Riemann invariants | xAct Invar / xTras | [xTras arXiv:1308.3493](https://arxiv.org/pdf/1308.3493) |

**[synthesis]** Architecturally: (1) monoterm-canonicalize every monomial; (2) represent the
expression as a vector over canonical monomials; (3) reduce modulo the linear span of multi-term
identities (row-reduced relation basis, generated on demand from Young symmetrizers or loaded from a
database). This gives a true normal form (a basis choice for the quotient space) and separates the
hard combinatorics (step 1) from linear algebra (step 3), which is also the easiest part to certify.

---

## 7. Fermions, spinors, gamma matrices, Wick contractions

- **Grassmann and spinor-metric signs** fit the signed-group model: xPerm and SymPy `tensor_can`
  support anticommuting factors and antisymmetric metrics (spinor ε: exchanging up/down in a dummy
  pair costs −1). Must be first-class in the new core, including the "zero by odd automorphism" test.
- **xAct extensions**: FieldsX (fermions, gauge fields, BRST) — [arXiv:2008.12422](https://arxiv.org/pdf/2008.12422).
- **Tensor–spinor canonicalization and Fierz expansion**: Gates, Hilsenrath, Hilsenrath, SusyPy
  (Cadabra module), 11D supergravity — [arXiv:2212.00614](https://arxiv.org/abs/2212.00614).
- **Gamma matrices / Clifford algebras in any dimension and signature**: Kuusela, GammaMaP —
  [arXiv:1905.00429](https://arxiv.org/pdf/1905.00429); Cadabra has built-in gamma algorithms.
  **[synthesis]** Clifford products are not a permutation-group problem: first rewrite to the
  antisymmetrised basis Γ^{a₁…a_k} (a basis of the bispinor space), whose coefficients are ordinary
  antisymmetric tensors, then feed them to the monoterm core. Traces and Fierz are algebraic
  preprocessing, not canonicalization.
- **Second quantization / Wick**: Wick&d (Evangelista) canonicalizes contractions *early* with an
  exhaustive combinatorial procedure — [arXiv:2205.01178](https://arxiv.org/html/2205.01178);
  SeQuant uses its graph canonicalizer to accelerate Wick's theorem (above); MobiDyck (Morere &
  Etienne, May 2026) uses Dyck-language structure to discard vanishing fermionic expectation values
  without applying Wick's theorem — [arXiv:2605.27159](https://arxiv.org/abs/2605.27159).

---

## 8. Feynman diagrams and Feynman-integral families

**Diagrams (deduplication and symmetry factors)** — canonical labelling + |Aut|:
- FeynGraph (Braun, ACAT 2025): Rust, ~10⁵ diagrams/s vs QGRAF ~10³–10⁴; topology generation then
  field assignment; symmetry factors; **diagram signs for multi-fermion operators** via the Denner
  flipping rule using connection information from UFO models (QGRAF models lack it) —
  [slides](https://indico.cern.ch/event/1488410/contributions/6597940/attachments/3133816/5560004/ACAT_25_FeynGraph.pdf),
  [GitHub](https://github.com/Jens-Braun/FeynGraph).
- GraphState (Batkovich, Kirienko, Kompaniets, Novikov): generalised **Nickel index** canonical
  labels for Feynman graphs — [arXiv:1409.8227](https://arxiv.org/abs/1409.8227).
- Background: Weinzierl, *Feynman Diagrams* review (2025) — [arXiv:2501.08354](https://arxiv.org/pdf/2501.08354);
  symmetry factors — [arXiv:2009.12616](https://arxiv.org/pdf/2009.12616); graphica generates
  diagrams from vertex signatures (Symbolica post).

**Integral families (topology mapping, sector symmetries)**:
- Pak's algorithm: canonical ordering of Feynman parameters via the characteristic polynomial —
  [arXiv:1111.0868](https://arxiv.org/pdf/1111.0868). Implemented in FIRE (tsort), Feynson
  ([GitHub](https://github.com/magv/feynson)), pySecDec.
- Jahn (pySecDec, PoS CORFU2017): graph-based matching **misses matroid symmetries**
  (non-isomorphic graphs with the same integral); polynomial canonicalization catches them —
  [PoS](https://pos.sissa.it/318/132/pdf).
- Going beyond constant momentum transformations (rational-function ansatz), 2024 —
  [arXiv:2406.20016](https://arxiv.org/pdf/2406.20016).
- Tool landscape (Reduze 2, Kira 3, FIRE 7, LiteRed) — review [arXiv:2510.10748](https://arxiv.org/html/2510.10748).

**[synthesis]** Pak's procedure is lexicographic canonicalization of a coefficient/exponent matrix
under column permutations — a canonical form of a coloured bipartite structure (monomials ↔
variables, edge colours = exponents, monomial colours = coefficients). It can therefore run on the
same IR engine, gaining its pruning instead of Pak's copy-and-sort enumeration.

---

## 9. Formal verification and certificates

| Work | Relevance | Link |
|---|---|---|
| Banković, Drecun, Marić, "A proof system for graph (non)-isomorphism verification" (LMCS 19(1), 2023) | McKay–Piperno canonical labelling as a proof system; implementation exports proofs that a graph is the canonical form; independent checker; rules, soundness and completeness formalised in **Isabelle/HOL** | [arXiv:2112.14303](https://arxiv.org/abs/2112.14303) |
| VeriPB + CakePB (formally verified checker); certified symmetry breaking with auxiliary-variable orders (2025) | end-to-end verified proof checking for symmetry reasoning | [arXiv:2511.16637](https://arxiv.org/pdf/2511.16637), [VeriPB docs](https://satcompetition.github.io/2025/downloads/checkers/veripb.pdf) |
| `permutation_factors` (Coq) | verified functional Schreier–Sims(–Minkwitz); small project | [GitHub](https://github.com/bergwerf/permutation_factors) |
| **`leanprover/hex-graph-iso`** (Apache-2.0, updated 2026-09-12; part of the `hex` Lean 4 computer-algebra project) | verified Lean 4 implementations of the pinned dense and sparse configurations of **nauty 2.9.3** for coloured simple undirected graphs; `graph_iso` tactic closing positive and negative isomorphism goals through the kernel; Mathlib-free, with a separate Mathlib bridge | [GitHub](https://github.com/leanprover/hex-graph-iso) |
| **`leanprover/hex-perm-group`** (Apache-2.0) | verified permutation-group algorithms for Lean 4 (stabilizer chains) | [GitHub](https://github.com/leanprover/hex-perm-group) |

*(The two `hex` repositories were found by the parallel study (§12) and confirmed via the GitHub API
and README during this survey.)*

**[synthesis] Certification strategy for the new core**:
1. *Orbit membership* (output is equivalent to input): emit the witness permutation and sign;
   checking is linear-time and trivial to verify in Lean.
2. *Canonicity* (equivalent inputs give identical output): the fastest route is now to **pin the
   core's canonical function to a configuration `hex-graph-iso` already verifies** (nauty 2.9.3
   semantics on coloured graphs), so the Lean side reuses an existing verified labeller instead of
   porting the Banković et al. proof system from Isabelle/HOL. Open: `hex-graph-iso` covers simple
   undirected coloured graphs, so the tensor encoding (bundles, hyperedges, multigraph diagrams) must
   be expressed in that class or the library extended.
3. *Zero by symmetry*: emit the odd automorphism as a witness; checkable directly. The automorphism
   group must be computed completely (not Monte Carlo) for this to be sound — see §12.
4. *Multi-term layer*: certificates are linear-algebra witnesses (relation coefficients), checkable
   by exact rational arithmetic.
No surveyed work formally verifies Butler–Portugal at production scale in Lean; the certificate and
pinned-configuration routes avoid needing to.

---

## 10. Recommended architecture **[synthesis]**

```
            ┌──────────────────── domain encoders ────────────────────┐
 tensor monomial │ diagram │ U/F polynomial │ fermionic TN │ Γ-basis coefficients
            └──────────────┬──────────────────────────────────────────┘
                           ▼
        coloured relational structure  +  signed-group gadgets
        (vertices: cores, slots, bundles, indices; colours; hyperedges;
         generator-given groups attached where cheaper than gadgets)
                           ▼
   fast paths: all-distinct factors → sort │ totally (anti)symmetric subsets → Niehoff-style
                           ▼
   IR / graph-backtracking canonical labeller  (refinement, individualization,
     automorphism pruning, orbit pruning; returns canon, witness, Aut generators)
                           ▼
   sign character on Aut generators → zero test;  witness → output sign
                           ▼
   certificate emitter (witness, IR proof, odd automorphism)      → Lean-checked checker
                           ▼
   multi-term layer: vector over canonical monomials, reduce modulo relation basis
```

Design rules:
- **Canonical form = deterministic function of the input**, independent of search order, threads, or
  which fast path fired. Fast paths must be proven (or certified) to return the *same* canonical
  representative as the general engine, or must only be used to *prune* the general engine.
- Keep the kernel allocation-free on warmed workspaces (as GraphCombinations.jl does); the realistic
  workload is millions of small instances.
- Benchmark baselines: xPerm (current), Niehoff's algorithm, SeQuant/bliss, Symbolica/graphica,
  Vole, GraphCombinations.jl, nauty/Traces; for diagrams, FeynGraph and QGRAF.
- Parallelism, racing and Pareto trade-offs: §12 (separate study).

---

## 11. Open questions (need experiments, not literature)

1. Gadget encoding vs generator-based groups: which slot-symmetry groups are cheaper as graph
   gadgets (S_n, A_n, Riemann pair symmetry) and which should stay as generators inside graph
   backtracking?
2. Does IR beat Niehoff-style BP on *single* highly symmetric tensors (e.g. a rank-10 totally
   antisymmetric form contracted with itself), not only on identical-factor products?
3. Spinor-metric and Grassmann signs inside IR: the sign character must be computed on
   automorphisms found during search; verify there is no interaction with orbit pruning that loses
   a sign-conflict (the xperm "zero" detection).
4. Canonical *index names* for output: SeQuant regenerates dummies in order of appearance, then
   optionally applies a cosmetic lexicographic sort; decide whether the user-visible form must be
   xAct-compatible (golden-master tests in `test/golden/`).
5. Pak-on-IR for integral families: validate against Feynson on known matroid-symmetry examples.
6. Certificate size and checking cost on realistic workloads (Banković et al. report per-instance
   proofs; unknown overhead at 10⁶ terms).

---

## 12. Parallelism, GPU, racing, Pareto

Full study (3,850 words, Pareto table, sources): [`08b_parallel_racing_pareto.md`](08b_parallel_racing_pareto.md),
produced by a separate agent. Its key arXiv citations were spot-checked against the arXiv API (dejavu parallel
2108.04590, sublinear IR traversal 2011.01726, FORM 5.0 2601.19982, GPU WL 2607.02603, hash consing 2509.20534,
concurrent hash tables 1601.04017, multithreaded FORM hep-ph/0702279) — all resolve to the described papers.
Quantitative bounds: [`09_canonicalization_perf_bound.md`](09_canonicalization_perf_bound.md).

**Findings.**
1. **Parallelize across terms, not within a term.** The workload is 10⁵–10⁷ independent small terms;
   TFORM (Tentyukov & Vermaseren, hep-ph/0702279) is the precedent (master/worker, chunked terms, tail
   stealing), and its serial final merge is the known bottleneck — collect with sharded hashing or
   radix sort instead.
2. **Determinism argument** (McKay & Piperno, arXiv:1301.1493, Lemma 4 / Thm 5a): automorphisms found by
   any thread in any order only prune; they never remove a leaf with the maximal node invariant, so the
   canonical form is independent of scheduling. Pruning by mere invariant *difference* is valid for
   groups, not for canonical forms.
3. **Monte Carlo symmetry search (dejavu, arXiv:2108.04590) gives groups, not canonical forms**, and can
   miss automorphisms. For signed/Grassmann terms a missed odd automorphism is a missed zero, making the
   output sign run-dependent — a correctness bug, not only a speed issue.
4. **A canonical form exists only for a pinned configuration**: the nauty 2.9.3 guide states the
   labelling changes between sparse/dense, nauty/Traces, digraph and invariant options. This dovetails
   with `hex-graph-iso` (§9), which verifies exactly such pinned configurations.
5. **GPU literature is thin**: randomized GPU colour refinement for single huge graphs
   (arXiv:2607.02603), a few large-graph GPU canonical-labelling papers with modest or unverified gains;
   nothing on tensor monomials or Feynman diagrams; FORM 5.0 has no GPU work.
6. **Racing must not let the winner choose the output.** Safe designs: all racers compute the same
   declaratively specified function (R1), racers contribute only automorphism generators to one
   deterministic search (R2), or one algorithm defines the form and others only test equality or are
   proven-equal fast paths (R3). Under a term-parallel load, per-instance algorithm selection with
   escalation after a time budget beats racing.
7. **Exact-arithmetic collection is order-sensitive**: checked `Rational{Int}` overflow can depend on
   summation order, so collection needs BigInt promotion or a fixed reduction tree.

**Pareto verdict** (08b §c): a term-parallel engine with proven-equal fast paths and a skeleton cache
(design C), plus deterministic intra-instance automorphism helpers for rare giant terms (design D),
dominates. First-finisher racing (F) and full-GPU canonization (I) are dominated.

**Reconciliation with the bound** **[synthesis]**: `09` shows a basic GPU has 3–15× lower *on-device*
floors than an 8-core CPU for low-search terms, but the PCIe floor erases most of that for offload, and
the win depends on an unmeasured SIMT efficiency with no supporting literature. Both analyses therefore
agree: CPU-first; a GPU is worth revisiting only for a pipeline where term generation, canonicalization
and collection all stay on the device.

---

## Appendix: reading order for implementation

1. xPerm paper (0803.0862) — the baseline you are replacing.
2. Niehoff (1702.08114) §3–4.4 — BP internals and exactly where it fails.
3. SeQuant (2511.09943) TN canonicalizer section — the concrete graph encoding with bundles/phases.
4. McKay & Piperno (1301.1493) — IR engine design.
5. Jefferson–Waldecker–Wilson (2209.02534) + Vole — canonical images under arbitrary groups.
6. Banković–Drecun–Marić (2112.14303) — certificate format and proof system.
7. Pak (1111.0868) + Jahn (PoS CORFU2017) — polynomial canonicalization for integral families.
8. Alakazam (2608.20452) and GraphCombinations.jl — the Julia competitors/building blocks.
