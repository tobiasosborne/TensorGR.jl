# Parallel canonicalization core: multicore, GPU, racing, Pareto

Scope: survey + design notes for a TensorGR.jl canonicalization core replacing xperm.c. Everything cited was retrieved during this survey. Where I reason past a source, it is marked **[own idea]**.

## Bottom line

1. **Work on many terms at once, and run each term on a single thread.** The realistic workload is 10^5–10^7 terms with at most about 100 slots each. Each term can be canonicalized independently, so hardware parallelism belongs at the term level. Within-instance parallel search is only for the rare giant instance.
2. **Parallel search inside one instance is safe for determinism only if it affects pruning and never chooses the output.** McKay–Piperno (Thm 5a) give the argument: any automorphisms, found by any thread in any order, only prune subtrees. They never remove the leaf with the best node invariant, so the canonical form is the same whatever the thread timing.
3. **Randomized, parallel symmetry search (dejavu) gives automorphism groups, not canonical forms.** Its authors say it is unclear whether their sublinear techniques carry over to canonization. It is also Monte Carlo with one-sided error (it can miss automorphisms). **[own idea]** For signed or Grassmann tensors, a missed odd automorphism means a missed zero, and then the sign of the output depends on the run. That is a determinism bug, not only a speed issue.
4. **GPU:** I found no literature on GPU canonicalization of tensor monomials or Feynman graphs. The GPU canonical-labeling work that exists targets a few large graphs, and the gains reported are modest or unverified. GPU color refinement (1-WL) now scales to huge single graphs, and it is randomized. Realistic GPU roles: batched invariant hashing as a prefilter, and sort/segmented-reduce for collecting terms. The core should be CPU-first.
5. **Racing must not let the winner choose the output.** Two designs are safe. Either race only the work that does not decide the output (automorphism generators, features), or race only exact algorithms that compute the *same specified* function (for example a minimal image under a fixed order). Of the portfolio methods, per-instance selection (SATzilla/AutoFolio style) suits cheap instances better than racing does.

---

## (a) Findings by topic

### 1. Multicore canonical labeling / automorphism search

- **Anders & Schweitzer, "Parallel Computation of Combinatorial Symmetries", ESA 2021, arXiv:2108.04590.** This is dejavu. They replace the sequential traversal of the individualization–refinement (IR) search tree with *random root-to-leaf walks*, which parallelize trivially. A probabilistic abort criterion uses sifting into a Schreier structure, and that sifting had to be made thread-safe because "sifting would often become the bottleneck".
  - Breadth-first work is shared through lock-free queues.
  - Results were competitive on 1 thread and beat Traces on 19 of 24 benchmark sets with 8 threads.
  - They note that prior practical parallel isomorphism work is "quite limited" (Tener's thesis).
  - Key caveats: it solves the **automorphism group** problem only, as Monte Carlo with *one-sided* error, meaning it may miss automorphisms but never reports false ones.
  - Subtlety: with many threads, elements that are fast to compute can bias the abort test. Their fix is to make all threads finish their current iteration when the counter trips.
- **dejavu 2.x (MIT; github.com/markusa4/dejavu; automorphisms.org).**
  - The README says it "computes the automorphism group"; it has no canonical labeling.
  - Each reported generator is certified by checking φ(G)=G. The docs say explicitly that there are no certificates for completeness (non-isomorphism), and that the preprocessor is trusted unless "strong certification" is switched on.
  - The current standalone option list (`--err`, `--gens`, seeding, `--permute`) shows no thread-count flag. I could not confirm whether multithreading survived into 2.x; this needs checking.
- **Anders & Schweitzer, "Search Problems in Trees with Symmetries", ICALP 2021, arXiv:2011.01726.** Randomized traversal visits roughly √(tree size) leaves for automorphisms and isomorphism. For canonization, they state it "is not clear to us whether any of the techniques for sublinear exploration developed in this paper can be transferred to the canonization problem". The reason is that the output leaf must be consistent across isomorphic inputs. The companion **ALENEX 2021 paper, arXiv:2011.09375**, avoids canonical labeling altogether for isomorphism testing.
- **McKay & Piperno, "Practical graph isomorphism, II", arXiv:1301.1493.** This paper supplies the determinism argument.
  - Canonical form: C(G,π₀) = G relabelled by *any* leaf achieving φ* = max φ over the leaves. It is independent of which maximizing leaf is chosen (Lemma 4).
  - **Theorem 5(a):** after *any* sequence of pruning operations P_A (invariant-based) and P_C (automorphism-based), at least one leaf with φ = φ* remains.
  - So automorphisms discovered in any order, by any thread, from random "experimental paths" (Traces uses random paths plus random Schreier), affect speed only and never the form.
  - Pruning P_B (≠) is valid for automorphism groups but *not* for canonical forms.
- **nauty & Traces User's Guide v2.9.3 (users.cecs.anu.edu.au/~bdm/nauty/nug29.pdf).**
  - The canonical labelling **changes** with the sparse vs dense representation, with Traces vs nauty, with the digraph option, and with invariant settings.
  - So "canonical form" means canonical for a fixed (algorithm, configuration, version). The new core must pin and version its canonical function.
  - nauty/Traces are thread-safe only when built with TLS (`--enable-tls`).
  - Its generators use `res/mod` splitting for embarrassingly parallel generation.
- **Verified conformance for pinned configurations: `leanprover/hex-graph-iso` (Apache-2.0).**
  - A Lean 4 verified coloured-graph canonical labelling whose pinned dense and sparse configurations are oracle-checked against nauty 2.9.3 (labels, canonical bits, visited-node counts).
  - It has certificates (`certifyKey?` / `checkCanon`).
  - The companion **`leanprover/hex-perm-group`** verifies stabilizer chains (`checkChain`).
  - This is highly relevant to the Lean plan, and to "the canonical function is a fixed spec".
- **Parallel Schreier–Sims / partition backtrack.** I found only Cooperman et al.'s memory- and disk-based algorithms for *very high degree* groups (ISSAC'03, ccs.neu.edu/home/gene/papers/issac03.pdf). That is irrelevant at ≤100 points. I found no practical parallel partition-backtrack work. **[own idea]** At degree ≤128, one Schreier–Sims run takes microseconds, so parallelizing within it is pointless; parallelize across instances.

### 2. GPU

- **Color refinement / 1-WL on GPU: Biondi, Tribastone & Tschaikowski, arXiv:2607.02603 (July 2026).**
  - A randomized refinement algorithm with probabilistic guarantees, plus correctness-preserving batching.
  - Up to about 100× faster than CPU partition refinement, and it handles graphs with over 30 billion edges.
  - It is **one massive graph**, not many small ones.
  - **[own idea]** A stable *partition* is isomorphism-invariant, but GPU color *names* must be assigned canonically (for example by hashing the multiset signature) before they can serve as a canonical hash.
- **GPU canonical labeling.**
  - **Wang, Guo, Ai, Li, Ren & Li**, "An Efficient Graph Isomorphism Algorithm Based on Canonical Labeling and Its Parallel Implementation on GPU", IEEE HPCC 2013. It claims 15–55× over CPU; I could only read the abstract.
  - **Son, Kim & Oh**, "An Efficient Parallel Algorithm for Graph Isomorphism on GPU using CUDA", IJET 2015. Divide-and-conquer blocks; gains are on graphs with tens of thousands of vertices.
  - **github.com/rana-dbouk/gpu-canonical-labeling.** nauty-style IR with tree parallelism (one thread block per branch) and node parallelism (refinement inside the block). No benchmarks were published.
- **Batched small-graph canonical codes on GPU: Kessl, Talukder, Anchuri & Zaki, "Parallel Graph Mining with GPUs", BIGMINE 2014 (PMLR v36).** GPU gSpan using minimum DFS codes. Speedup is at most 9× over sequential, and they warn that other groups' reported GPU speedups "may be too optimistic".
- **Nothing found** on GPU canonicalization of tensor expressions, Feynman diagrams, or permutation groups (Schreier–Sims / orbits). **FORM 5.0 (arXiv:2601.19982)** adds a GRACE-based diagram generator and TFORM improvements, but no GPU work.
- **Realism [own idea].**
  - IR search on a 20–100 vertex graph is short, branchy, and data-dependent (backtracking with variable depth). That is the worst case for SIMT warps.
  - On CPU, a ≤128-vertex adjacency fits in 2×64-bit words per row (nauty's dense mode already uses word-sized bitsets, WORDSIZE up to 128). So SIMD/bitset refinement is fast and cache-resident.
  - GPU fits the *regular* stages: a fixed number of WL rounds over a batch (block-diagonal CSR), hashing, radix sort, segmented reduction of coefficients.
  - Expected win: prefiltering and collection when there are 10^7+ terms and the data is already on the device. Not the canonicalization itself.

### 3. Throughput-level parallelism

- **TFORM: Tentyukov & Vermaseren, "The Multithreaded version of FORM", hep-ph/0702279 (CPC 2010).** This is the closest prior art for "10^6 terms in parallel".
  - A master thread plus workers, with terms distributed in chunks. Workers own private memory.
  - Load balancing works by stealing back the tail of a busy worker's chunk.
  - The known bottleneck is the **final merge-sort in a single thread**, "one compare per term". Workers write directly into the master's sort buffers.
  - Lesson: do the collection as a parallel sort or sharded hash, not a central merge.
- **Maier, Sanders & Dementiev, "Concurrent Hash Tables: Fast and General(?)!", arXiv:1601.04017 (PPoPP 2016).** Lock-free linear-probing tables that scale, including growable variants. Most libraries fall short under contention and resizing. Relevant when many terms share one canonical key (heavy merging).
- **Hash consing in JuliaSymbolics: Zhu et al., arXiv:2509.20534 (2025).**
  - A global weak-reference table; up to 3.2× compute speedup and 2× less memory.
  - Workloads with few duplicates show "slight overhead".
  - For multithreading they use *task-local* caches ("lock-free operation"), accepting duplicated work across threads. Cross-thread caching is left as future work.
- **Jefferson, Jonauskyte, Pfeiffer & Waldecker, "Minimal and canonical images", arXiv:1703.00197 (J. Algebra 2019).** Splitting |X| objects into orbits takes |X| canonical-image computations followed by a sort or hash. That is exactly term collection.
- **Invariant prefilters [own idea, grounded in McKay–Piperno's definition of a node invariant].**
  - A cheap invariant hash (refined color signature) can only *separate* terms. So terms whose invariant bucket has size 1 can never merge with another term, and could skip full canonization for the purpose of collection.
  - **But:** (i) zero detection (T = −T under an odd automorphism) still needs the automorphism group; (ii) output must be canonical for later cross-expression comparisons.
  - So the skip is safe only for "collect" and only when (ii) is not needed.
  - The safe universal fast path: **if color refinement yields a discrete partition, the automorphism group is trivial, the labeling is canonical with no search, and there is no zero or sign ambiguity.** dejavu similarly terminates "deterministically" when refinement settles everything.
- **Memoization [own idea].** Canonical forms of products do not compose from the canonical forms of their factors, because dummy contractions couple them. What caches well is the *skeleton*: the contraction graph with the tensor-name coloring, ignoring free-index names. Canonize the skeleton once, together with its automorphism group and orbit data, then apply it to every term sharing that skeleton. Also memoize the stabilizer chain (strong generating set) of each slot-symmetry group per tensor-name tuple. Cache design: task-local with per-batch merge, following the Symbolics.jl result.

### 4. Racing / portfolios

- **Gomes & Selman, "Algorithm portfolios", Artificial Intelligence 126 (2001).** Portfolios pay off when runtimes are heavy-tailed.
- **Xu, Hutter, Hoos & Leyton-Brown, "SATzilla", JAIR 32 (2008), arXiv:1111.2249.** Per-instance selection using empirical hardness models.
- **Lindauer, Hoos, Hutter & Schaub, "AutoFolio", JAIR 53 (2015).** An automatically configured selector.
- **Kotthoff's survey, arXiv:1210.7959.**
- **Kotthoff, McCreesh & Solnon, "Portfolios of Subgraph Isomorphism Algorithms", LION 2016 (doi:10.1007/978-3-319-50349-3_8).** Per-instance selection with graph features gives substantial wins. This is the closest graph analogue.
- **Balyo, Sanders & Sinz, HordeSat, arXiv:1505.03340.** A massively parallel portfolio with clause sharing.
- **Hamadi, Jabbour, Piette & Sais, "Deterministic Parallel DPLL", JSAT 7 (2011).** Parallel portfolios are non-reproducible because of weak synchronization. Their fix (synchronization barriers) keeps portfolio performance while making results fully reproducible. This is the direct precedent for our determinism requirement.
- **The core subtlety for canonization.**
  - Different exact algorithms generally output *different* representatives: nauty's options change the output even within one tool.
  - "First finisher wins" is therefore non-deterministic unless all racers compute the same function.
  - Three safe designs:
    - **(R1) Same spec.** Define the canonical form declaratively, for example the lexicographically *minimal image* of the slot configuration under the group generated by slot symmetries and dummy relabelings, with a fixed total order. Any correct algorithm (group backtrack, graph-based search, brute force for tiny cases) then returns identical output. Cost: minimal images are harder than "canonical images", which is the reason Jefferson et al. introduced the relaxed notion. **[own recollection, to verify in arXiv:0803.0862]** Butler–Portugal/xPerm's representative is a double-coset minimum, which is exactly this kind of spec.
    - **(R2) Race the group, fix the final step.** Racers (random walks à la dejavu, the Butler–Portugal group engine, graph search via bliss, dejavu, etc.) only contribute *automorphism generators* to a shared, deterministic search. By McKay–Piperno Thm 5(a), the result cannot depend on which racer won.
    - **(R3) One defining algorithm.** One algorithm defines the form; the others are used only for "same or different" checks (collection) or as fast paths that are *proven* equal to the defining algorithm on their domain. An example is a discrete-refinement fast path that is literally the first step of the defining algorithm.
- **[own idea]** Under a saturated term-parallel workload, racing burns cores that would otherwise process other terms. Use selection (cheap features: slot count, symmetry-group order, number of identical factors, refinement discreteness) plus *escalation* (start with the cheapest exact method and switch after a budget), and race only the rare long tail.

### 5. Language / runtime (Julia vs Rust/C core)

- **Julia threading.** Julia's scheduler is depth-first, based on partr (Bezanson, Nash & Pamnany, julialang.org blog, 2019). That post shows spawn-heavy code multiplying allocations (76 MiB → 687 MiB in their mergesort example). The takeaway is to chunk work, not spawn per term. The post predates later runtime changes, so re-check current behaviour.
- **GPU in Julia.** KernelAbstractions.jl gives vendor-neutral kernels (JuliaHub blog). KernelForge.jl (arXiv:2603.18695) reports scan and mapreduce times matching CUB on an A40. So the GPU prefilter and collection stages could be written in Julia.
- **FFI thread safety.** nauty requires a TLS build. The current xperm wrapper serializes only `dlopen`. **[own idea]**
  - Any C or Rust core must be reentrant and must receive **batches**: a flat array of term encodings per call, which amortizes FFI cost and allows SIMD.
  - Avoid two competing thread pools (Julia threads and rayon). Either Julia owns parallelism and calls a single-threaded core per chunk, or the core owns a pool and Julia makes one blocking call.
  - Rust prior art: Symbolica/graphica (main thread covers it).

---

## (b) Recommended parallel architecture [own idea, built on the sources above]

```
terms (10^5–10^7) ──► [S0] encode: flat SoA, per-term bitset graph (≤128 slots), tensor-name colors
                  ──► [S1] parallel over fixed-size chunks (deterministic chunking):
                          refine → canonical color names → invariant hash h(t)
                          if partition discrete: canonical form immediately (trivial Aut, no zero)
                  ──► [S2] per-term exact canonicalizer, ONE canonical spec, single-threaded per term:
                          tier 0: lookup/brute force for tiny cases (same spec, cross-checked in CI)
                          tier 1: Niehoff-style special cases (proven equal to spec on their domain)
                          tier 2: main engine (group-coset or graph IR; ONE defining choice), skeleton cache
                          budget exceeded ──► [S3]
                  ──► [S3] hard-instance mode (rare): deterministic master search owns φ* and output;
                          helper threads run seeded random walks / alt engines → automorphism
                          generators → shared Schreier structure used ONLY for P_C pruning
                  ──► [S4] collection: shard by hash(canonical key) mod P; per-shard local hash map
                          (or radix sort + segmented reduce); exact-rational coefficient sums;
                          final output sorted by canonical key
                  ──► [S5] optional certificates per term (embarrassingly parallel emit + check)
```

**Determinism argument.**

- **D1. A single pinned function.** The canonical form is a fixed function f(term): one refinement, one cell selector, one node invariant φ or one minimal-image order, and a versioned spec. nauty shows how changing any of these changes the output. Tiers 0 and 1 are admissible only when they provably compute f. Enforce this with property tests against tier 2, and eventually with Lean lemmas.
- **D2. Parallelism inside an instance affects pruning only.** In S3, helpers contribute only automorphisms (checked by g(G)=G before use), and those are used only for P_C pruning. By McKay–Piperno Thm 5(a), a leaf with the maximal φ survives any such sequence of prunings, and C is independent of which maximal leaf is chosen. So the output does not depend on the schedule. Seeds and PRNGs live only in the helpers.
- **D3. Signs and zeros need the full group.** For signed or Grassmann terms, the output is ±C or 0. If the canonical *labeling* could be off by an automorphism of sign −1, that would mean the true value is 0. So zero detection needs a *complete* generating set of Aut, not a Monte Carlo one. Deterministic exhaustive search supplies this: by Thm 5(b), the P_C generators together with the remaining max-leaves generate Aut. Then check the sign character on the generators. **Never let a probabilistic abort (dejavu-style) decide the sign or zero status.**
- **D4. Collection is order-independent.**
  - Keys are canonical forms. Exact rational addition is associative and commutative. The final sort by key removes any dependence on insertion order.
  - Caveat: checked `Rational{Int}` overflow can happen in some summation orders and not in others. Use overflow-to-BigInt promotion or a fixed reduction tree.
  - Floating-point coefficients would need a fixed reduction order.
- **D5. Racing only in the D2 sense or the R1 sense.** Racers either feed automorphisms (R2) or compute the identical spec (R1). A winning racer never chooses the representative.

---

## (c) Pareto table (qualitative; ✓ good, ~ medium, ✗ poor)

| Design point | Latency, small term | Throughput 10^6 small | Worst case (big symmetric) | Deterministic | Cert/proof cost | Memory | Impl. complexity |
|---|---|---|---|---|---|---|---|
| A. Term-parallel, single-threaded group engine (xperm-like double coset) | ✓ | ✓ (scales with cores) | ~ (Schreier–Sims fine at ≤128; dummies can blow up) | ✓ | ~ (coset/SGS certificates) | ✓ | ~ |
| B. Term-parallel graph IR (nauty/bliss-like, pinned config) | ~ (graph build overhead) | ✓ | ✓ (automorphism pruning) | ✓ if pinned | ✓ (known proof systems: Banković et al.; hex-graph-iso) | ~ | ✗ (unless reused) |
| C. A or B + fast paths (discrete refinement, Niehoff, tiny brute force) + skeleton cache | ✓✓ | ✓✓ | inherits A/B | ✓ if fast paths ≡ spec | ✓ (fast paths have trivial certificates) | ~ (cache) | ~ |
| D. C + S3 intra-instance helpers (automorphisms only) | same as C | same as C | ✓✓ | ✓ (Thm 5a) | ~ | ~ | ✗ |
| E. dejavu-style Monte Carlo, group only | n/a | n/a | ✓✓ for Aut | ✗ for canonical forms and signs | generators only; no completeness proof | ✓ | ~ (reuse dejavu, MIT) |
| F. Racing, first-finisher output (different specs) | ✓ | ✗ (steals cores) | ✓ | ✗ | ✗ | ✗ | ~ |
| G. Racing with fixed final step (R2) or same spec (R1) | ✓ | ~ | ✓ | ✓ | ~ | ~ | ✗ |
| H. GPU batched WL prefilter + GPU sort/reduce collection + CPU canonization | ~ (transfer) | ✓ at ≥10^7 terms (unmeasured) | no help | ✓ if colors named canonically | none (prefilter is untrusted) | ✗ (device memory) | ✗ |
| I. Full canonization on GPU | ✗ | ? (no evidence) | ✗ | ~ | ✗ | ✗ | ✗✗ |
| J. Minimal-image spec (R1), any exact algorithm | ✗ for large groups | ~ | ✗ | ✓✓ (algorithm-independent) | ✓ (easy statement for Lean) | ✓ | ~ |

**Dominance for the realistic workload (many small terms, occasional giant term):**

- **C + D** dominates. C wins the bulk, and D fixes the tail without breaking determinism.
- **G** is dominated by selection + escalation, except for rare hard instances, where it collapses to D.
- **F and I** are dominated.
- **E** is a component (a helper inside D), not a design point.
- **H** is optional later, once profiling shows collection or hashing is the bottleneck at 10^7 terms.
- **Choosing A or B as the defining engine** is the main open decision for the core. B generalizes to Feynman graphs, tensor networks, and topology mapping, and has existing verification infrastructure (hex-graph-iso, isocert). A is tighter for pure slot symmetries and for metric or ε dummies. A hybrid is possible: B builds the graph, with the tensor slot groups encoded as gadgets.

---

## (d) Open questions / what to benchmark

1. **Fraction of terms settled by discrete refinement** (fast path) on the real workloads: the bench_12 six-derivative dS terms, xAct golden masters, and Riemann^n monomials. This number sets the value of the whole design.
2. **A vs B per-term latency** at 10–100 slots, including graph-construction cost. Include the metric- or ε-dummy sign cases.
3. **Skeleton-cache hit rate**: how many distinct skeletons per 10^6 terms?
4. **Collection:** sharded local hash maps vs radix sort + segmented reduce vs a Maier–Sanders–Dementiev-style global lock-free table, on 1–64 threads. Watch the TFORM-style single-thread merge bottleneck and Julia GC pressure.
5. **The S3 threshold:** at what instance size (for example products of k identical Riemanns, 6-loop vacuum graphs) does intra-instance parallelism beat the term-parallel baseline?
6. **Check whether dejavu 2.x still supports threads**, and whether its C++ API can serve as an S3 automorphism helper, given a pinned version and a strong-certification mode.
7. **Certificate overhead** per term (emit + check) as a fraction of canonicalization time. Is it cheap enough to keep always on in CI mode?
8. **GPU** (only after items 1–4): batched fixed-round WL hashing + sort on 10^7 terms, compared with the CPU pipeline end to end, including transfers.
9. **Spec question:** can the canonical function be defined declaratively (minimal image) at acceptable cost for ≤100 slots? If yes, R1 racing and a Lean statement both become simple.
10. **FFI and threading:** one pool (Julia owns it) vs a core-owned pool; the cost of the batch-call API vs per-term ccall.

## Sources (all retrieved)

- Anders & Schweitzer, ESA 2021, arXiv:2108.04590; ICALP 2021, arXiv:2011.01726; ALENEX 2021, arXiv:2011.09375; dejavu README/DOCS (github.com/markusa4/dejavu) and automorphisms.org standalone + certification pages.
- McKay & Piperno, arXiv:1301.1493; nauty & Traces User's Guide v2.9.3 (users.cecs.anu.edu.au/~bdm/nauty/nug29.pdf).
- leanprover/hex-graph-iso and leanprover/hex-perm-group (GitHub).
- Cooperman et al., ISSAC'03 (ccs.neu.edu/home/gene/papers/issac03.pdf), search listing only.
- Biondi, Tribastone & Tschaikowski, arXiv:2607.02603.
- Wang et al., IEEE HPCC 2013 (ieeexplore 6832036; abstract via Semantic Scholar only).
- Son, Kim & Oh, IJET 2015 (hgpu.org/?p=15037).
- github.com/rana-dbouk/gpu-canonical-labeling.
- Kessl et al., BIGMINE 2014 (proceedings.mlr.press/v36/kessl14.pdf).
- FORM 5.0, arXiv:2601.19982.
- Tentyukov & Vermaseren, hep-ph/0702279.
- Maier, Sanders & Dementiev, arXiv:1601.04017.
- Zhu et al., arXiv:2509.20534.
- Jefferson, Jonauskyte, Pfeiffer & Waldecker, arXiv:1703.00197.
- Gomes & Selman, AI 126 (2001) (sciencedirect S0004370200000813).
- Xu et al., arXiv:1111.2249.
- Lindauer et al., JAIR 53 (2015).
- Kotthoff, arXiv:1210.7959.
- Kotthoff, McCreesh & Solnon, LION 2016 (doi:10.1007/978-3-319-50349-3_8).
- Balyo, Sanders & Sinz, arXiv:1505.03340.
- Hamadi et al., JSAT 7 (2011) (content.iospress.com … sat190081).
- Bezanson, Nash & Pamnany, julialang.org/blog/2019/07/multithreading.
- JuliaHub KernelAbstractions blog.
- Pilliat, arXiv:2603.18695.
