# TInvar Design: Tensorial Invariant Canonicalization

Research output for TGR-5lp.1.

## 1. Problem Statement

The existing Invar pipeline (`riemann_simplify`, levels 1--6) handles **scalar**
Riemann invariants: products of curvature tensors with all indices contracted
by metrics. The contraction pattern is encoded as a fixed-point-free involution
on 4k slots (`RInv`), and canonicalization finds the lex-smallest involution in
the orbit of the combined Riemann slot-symmetry and inter-factor permutation
group acting by conjugation.

**TInvar** extends this to **tensorial** Riemann monomials: products of
curvature tensors where some indices remain free (uncontracted). Examples:

- `R_{abcd} R^{ab}_{ef}` -- rank-4 tensor, free indices c, d, e, f
- `R_{a}^{bcd} R_{bcde}` -- rank-2 tensor, free indices a, e
- `nabla_a R_{bcde}` -- rank-5 tensor (differential tensorial monomial)

The challenge is threefold:

1. **Representation**: RInv encodes a fixed-point-free involution. With free
   indices, some slots have no partner -- the involution must allow fixed
   points (or the representation must separate free slots).

2. **Canonicalization**: The symmetry group acting on a tensorial monomial is
   smaller than the scalar case, because free indices cannot be freely
   permuted among each other. The canonical form must respect the
   *symmetry type* of the free indices (e.g., which irrep of S_r they
   belong to, where r is the rank).

3. **Relations**: Bianchi identities and DDIs generate linear relations
   among tensorial monomials of the same free-index structure. The
   independent basis depends on both the contraction pattern and the
   symmetry type of the free indices.

## 2. Background: How xperm Handles Free vs Dummy Indices

The Butler-Portugal algorithm in xperm.c (`canonical_perm_ext`) already
makes a sharp distinction between free and dummy indices:

### The Two-Step Algorithm

1. **`coset_rep` (free-index canonicalization)**: Finds an element `s` of the
   slot symmetry group S such that `s * g` places the free indices in their
   canonical (lexicographically smallest) slot positions. After this step,
   the SGS is restricted to the **stabilizer** of those free slots.

2. **`double_coset_rep` (dummy-index canonicalization)**: With free slots
   pinned, finds the canonical representative of the double coset
   `S_stab \ g' / D`, where D is the dummy-index symmetry group (pair
   exchanges and metric symmetry).

This is precisely the mathematical structure needed for tensorial invariants:
the free-index slots define a **coset** problem (they break the full
symmetry group), and the remaining dummy-index slots define a
**double-coset** problem.

### Key Observation

The existing `canonicalize.jl` in TensorGR uses an **all-free** mode (lines
213--228), treating every index as free and handling dummy normalization
separately in `_normalize_dummies`. This sidesteps the double-coset
algorithm entirely. For TInvar, we need to properly separate the two
concerns:

- **Genuinely free** indices: canonicalized via `coset_rep` (Step 1).
- **Dummy** indices: canonicalized via `double_coset_rep` (Step 2).

The existing `xperm_canonical_perm_ext` wrapper already accepts both
`free_names` and `dummy_names` and calls `canonical_perm_ext` directly
in Renato's notation. This is the correct entry point for TInvar.

## 3. Proposed TRInv Type

### Design Choice: Partial Involution (Option B)

After analyzing the three options from the task description:

- **Option A (RInv + free_slots)**: Simple but duplicates information --
  the contraction vector already contains enough data if we allow fixed
  points (or a sentinel value).

- **Option B (Partial involution)**: A contraction map where free slots
  map to themselves (fixed points). This is the most natural generalization:
  `sigma(i) = i` means "slot i is free", `sigma(i) = j != i` means "slot i
  contracts with slot j". The involution property `sigma(sigma(i)) = i`
  still holds (a fixed point is its own inverse).

- **Option C (Separate struct)**: Unnecessary -- the scalar RInv is
  precisely the special case of TRInv where there are no fixed points.

**Recommendation: Option B** -- a partial involution where fixed points
denote free slots. This subsumes RInv (which requires no fixed points).

### Type Definition

```julia
"""
    TRInv(degree, contraction, free_slots)

Contraction permutation representation for a tensorial Riemann monomial.

For degree `k` (k Riemann factors), the contraction is an involution of
length `4k`: `contraction[i]` gives the slot paired with slot `i`, or `i`
itself if slot `i` is free.

The free_slots field explicitly lists the positions of free indices for
fast access. This is derivable from the contraction (the fixed points),
but stored separately to avoid recomputation.

# Fields
- `degree::Int` -- number of Riemann factors
- `contraction::Vector{Int}` -- involution (length 4k), fixed points = free
- `free_slots::Vector{Int}` -- sorted list of free slot positions
- `free_positions::Vector{IndexPosition}` -- Up/Down for each free slot
- `canonical::Bool` -- true if in canonical form
"""
struct TRInv
    degree::Int
    contraction::Vector{Int}
    free_slots::Vector{Int}
    free_positions::Vector{IndexPosition}
    canonical::Bool
end
```

### Why Store `free_positions`?

In the scalar case, index positions (Up/Down) are irrelevant because all
contractions are through the metric `g^{ab}`, which pairs one Up with one
Down. For tensorial monomials, the free indices carry meaningful position
information: `R_{abcd}` with free `c` Down is a different object from
`R_{ab}{}^{c}{}_{d}` with free `c` Up. The canonical form must preserve
this distinction.

### Relationship to RInv

An `RInv` is a `TRInv` with empty `free_slots`:

```julia
# Conversion
TRInv(rinv::RInv) = TRInv(rinv.degree, rinv.contraction, Int[],
                           IndexPosition[], rinv.canonical)

# Validation
is_scalar(trinv::TRInv) = isempty(trinv.free_slots)
rank(trinv::TRInv) = length(trinv.free_slots)
```

### Constructor Validation

```julia
function TRInv(degree::Int, contraction::Vector{Int},
               free_slots::Vector{Int}, free_positions::Vector{IndexPosition})
    n = 4 * degree
    length(contraction) == n ||
        error("TRInv: contraction length != 4*degree")
    length(free_slots) == length(free_positions) ||
        error("TRInv: free_slots and free_positions must have same length")

    # Validate: involution (with fixed points allowed)
    for i in 1:n
        ci = contraction[i]
        (1 <= ci <= n) || error("TRInv: contraction[$i]=$ci out of range")
        contraction[ci] == i ||
            error("TRInv: not an involution at $i")
    end

    # Validate: free_slots are exactly the fixed points
    actual_free = sort([i for i in 1:n if contraction[i] == i])
    sort(free_slots) == actual_free ||
        error("TRInv: free_slots do not match fixed points of contraction")

    TRInv(degree, contraction, sort(free_slots), free_positions, false)
end
```

## 4. Canonicalization Algorithm

### Overview

Tensorial canonicalization uses xperm's two-step algorithm directly,
via `xperm_canonical_perm_ext`. This is **not** a BFS orbit enumeration
(which would be infeasible for high-rank tensorial monomials), but the
polynomial-time Butler-Portugal algorithm.

### Algorithm

```
canonicalize(trinv::TRInv) -> TRInv
```

1. **Build symmetry generators** (`_trinv_slot_generators`):
   - Per-factor Riemann symmetries (antisymmetry in pairs, pair swap) --
     same as `_rinv_slot_generators`.
   - Inter-factor transpositions (identical Riemann tensors may swap
     their 4-slot blocks).
   - **Important**: free indices break inter-factor symmetry. Two
     factors can only be exchanged if their free-slot patterns are
     compatible (same set of free/contracted positions within the block).

2. **Classify indices**:
   - Free slots: those with `contraction[i] == i`.
   - Dummy pairs: those with `contraction[i] != i`, grouped as
     (i, contraction[i]) with i < contraction[i).

3. **Assign names** (the xperm "name" convention):
   - Dummy pairs get consecutive names: pair k gets (2k-1, 2k).
   - Free indices get names after all dummies.

4. **Build the xperm permutation** (slot -> name map):
   - For dummy slot i paired with slot j (i < j): slot i gets the "up"
     name, slot j gets the "down" name of the pair.
   - For free slot i: gets its free-index name.

5. **Call `xperm_canonical_perm_ext`** with:
   - The permutation (slot -> name)
   - The symmetry generators (conjugated as in `_canonicalize_product`)
   - `free_names`: the names assigned to free slots
   - `dummy_names`: the names assigned to dummy pairs, as [u1,d1,u2,d2,...]

6. **Reconstruct TRInv** from the canonical permutation:
   - Extract the new contraction pattern from the canonical name
     assignments.
   - Extract sign from the last two points.
   - If sign is negative, the monomial acquires a factor of -1.

### Key Difference from BFS Orbit Enumeration

The existing `canonicalize(::RInv)` uses BFS orbit enumeration under
**conjugation** (`sigma -> g . sigma . g^{-1}`). This is correct for
RInv but has exponential worst-case complexity. For TRInv, the problem
is different: we are not canonicalizing the contraction under conjugation,
but canonicalizing the **index assignment** under the slot symmetry group
S (left action) and dummy symmetry group D (right action). This is
precisely the double-coset problem that xperm solves in polynomial time.

The RInv conjugation problem can also be reformulated as a double-coset
problem (Garcia-Parrado & Martin-Garcia 2007, Sec 3.2), but the current
RInv implementation does not do this. TInvar should use xperm from the
start, both for correctness and scalability.

### Handling Inter-Factor Symmetry with Free Indices

When all indices are contracted (scalar case), any two Riemann factors
can be exchanged because they are structurally identical. With free
indices, two factors can only be exchanged if they have the **same
free-slot pattern**: the same positions (within their 4-slot block) are
free, and those free positions have the same Up/Down configuration.

Example: In `R_{abcd} R^{ab}_{ef}`, factor 1 has all indices contracted
with something, and factor 2 has slots 3,4 free. These factors have
different free-slot patterns and cannot be interchanged.

In `R_{abcd} R_{ef}^{ab}`, both factors have the same free-slot pattern
(slots 1,2 free in the block), so they *can* be exchanged (with appropriate
sign from the antisymmetries).

The generator construction must check this:

```julia
function _trinv_factor_exchangeable(trinv::TRInv, f1::Int, f2::Int)
    off1, off2 = 4(f1-1), 4(f2-1)
    for j in 1:4
        is_free_1 = trinv.contraction[off1+j] == off1+j
        is_free_2 = trinv.contraction[off2+j] == off2+j
        is_free_1 == is_free_2 || return false
        if is_free_1
            # Also check positions match
            pos1 = trinv.free_positions[findfirst(==(off1+j), trinv.free_slots)]
            pos2 = trinv.free_positions[findfirst(==(off2+j), trinv.free_slots)]
            pos1 == pos2 || return false
        end
    end
    true
end
```

## 5. Enumeration of Independent Tensorial Monomials

### Problem

Given degree k and a free-index structure (specified by rank r and the
symmetry type of the free indices), enumerate all independent tensorial
monomials.

### Slot Budget

A degree-k monomial has 4k index slots total. If rank is r (r free
indices), then 4k - r slots must be contracted in pairs, requiring
(4k - r)/2 dummy pairs. This means 4k - r must be even, i.e.,
r must have the same parity as 4k.

### Enumeration Algorithm

```
enumerate_tensorial_monomials(k, free_structure) -> Vector{TRInv}
```

1. **Choose which slots are free**: Select r slots out of 4k to be free.
   The free slots must be assigned Up/Down positions. Not all choices
   are independent -- Riemann symmetries identify many of them.

2. **Choose the contraction pattern for the remaining slots**: This is
   a perfect matching (fixed-point-free involution) on 4k - r points.

3. **Canonicalize** each candidate via `canonicalize(::TRInv)`.

4. **Deduplicate**: Remove duplicates (same canonical form) and zero
   monomials (sign cancellation).

### Practical Strategy: Reuse `all_contractions`

The existing `all_contractions` function in `ansatz.jl` already solves a
closely related problem: given a list of tensors and a set of free
indices, enumerate all independent contractions. The TInvar enumeration
can be implemented as:

```julia
function enumerate_tinvar_monomials(degree::Int, free_idxs::Vector{TIndex};
                                     registry=current_registry())
    # Build k copies of the Riemann tensor with distinct dummy indices
    riem_copies = [Tensor(:Riem, [down(Symbol(:r, f, :_, j))
                   for j in 1:4]) for f in 1:degree]

    # Use all_contractions to enumerate independent matchings
    contractions = all_contractions(riem_copies, free_idxs)

    # Convert each to TRInv representation
    [from_tensor_expr_tensorial(c; registry=registry) for c in contractions]
end
```

This leverages the existing perfect-matching + canonicalize + dedup
infrastructure. The `from_tensor_expr_tensorial` function would generalize
`from_tensor_expr` (currently in `rinv.jl`) to handle expressions with
free indices.

### Independent Counts (Ground Truth)

For reference, the numbers of independent Riemann monomials at low
degrees, for various symmetry types of the free indices:

**Degree 1 (single Riemann), d >= 4:**
- Rank 0 (scalar): 0 (single Riemann has no scalar contractions by
  tracelessness of the Weyl part; only R via double trace)
- Rank 2 (symmetric): R_{(ab)} = R_{ab} (Ricci, 1 independent)
- Rank 2 (antisymmetric): 0 (R_{[ab]} = 0 by Ricci symmetry)
- Rank 4 (Riemann symmetry): R_{abcd} (1 independent, up to symmetries)

**Degree 2 (two Riemanns), d >= 4:**
- Rank 0 (scalar): 3 independent (R^2, R_{ab}R^{ab}, R_{abcd}R^{abcd})
- Rank 2: multiple, depends on symmetry type of free pair

### Symmetry Type of Free Indices

The free indices of a tensorial monomial carry a definite symmetry type
under permutations. For rank-2 tensors, the symmetry type is one of:
- Symmetric: T_{(ab)}
- Antisymmetric: T_{[ab]}
- No definite symmetry: T_{ab} (decomposes into symmetric + antisymmetric)

For higher rank, the symmetry type is specified by a Young diagram. The
enumeration should accept a Young diagram as input and project the result
using `young_symmetrize` from `src/algebra/young.jl`.

In general, for a given degree k and target Young diagram lambda for the
free indices, the number of independent tensorial monomials is:

```
N(k, lambda, d) = dim(orbit space of partial involutions
                       under Riemann symmetries and lambda-projection)
```

This is computed by the enumerate-canonicalize-dedup algorithm above,
optionally followed by Young projection to extract monomials with a
specific free-index symmetry.

## 6. Relation Generation: Bianchi and DDI in the Tensorial Case

### First Bianchi Identity (Cyclic Symmetry)

The first Bianchi identity R_{a[bcd]} = 0 is a **tensorial** identity
(all indices are free). When applied to a monomial of degree k with
free indices, it generates relations between TRInv elements with the
same free-index structure.

The key insight is that the Bianchi identity acts **within a single
Riemann factor**: it cyclically permutes three of the four indices of
one factor while leaving all other factors and the contraction pattern
unchanged. For a degree-k monomial with factor f having slots
(off+1, off+2, off+3, off+4):

```
R_{a b c d} = -R_{a c d b} - R_{a d b c}
```

This generates two new TRInv's from each original one by permuting
slots (off+2, off+3, off+4) cyclically. If any of the permuted slots
are contracted with other slots, the contraction pattern changes
accordingly.

**Implementation**: Apply `apply_bianchi_cyclic` (from
`simplify_levels.jl`) to each Riemann factor of each monomial, then
canonicalize and collect terms. The resulting linear relations reduce
the independent basis.

### Second Bianchi Identity (Differential)

The second Bianchi identity `nabla_{[a} R_{bc]de} = 0` is relevant for
**differential** tensorial monomials (those containing covariant
derivatives of curvature). For non-differential monomials (pure products
of undifferentiated Riemann tensors), the second Bianchi does not apply.

For differential tensorial monomials, the second Bianchi generates
relations of the form:

```
nabla_a R_{bcde} + nabla_b R_{cade} + nabla_c R_{abde} = 0
```

This is a tensorial identity with 5 free indices. When contracted with
other tensors, it produces relations between differential tensorial
monomials.

### Dimensionally-Dependent Identities (DDIs)

DDIs come from `delta^{a_1...a_{d+1}}_{b_1...b_{d+1}} = 0`. In the
scalar case, contracting all upper and lower indices of this generalized
delta with curvature tensors produces scalar identities (Gauss-Bonnet,
etc.).

For tensorial monomials, **not all indices of the generalized delta need
to be contracted with curvature tensors**. Some can remain free,
generating **tensorial DDIs**. For example:

In d=4, the vanishing of `delta^{abcde}_{fghij}` contracted with one
Riemann tensor and two metrics produces a rank-2 identity:

```
R_{ab} - (R/d) g_{ab} + ... = 0  (in low d)
```

In d=3, the Weyl tensor vanishes identically -- this is a tensorial DDI
(rank-4 identity with Riemann symmetry).

**Implementation**: The existing DDI machinery in `ddi_rules.jl` can be
extended to generate tensorial DDI rules. The generalization is:

1. Start with the generalized delta identity (order d+1).
2. Contract some pairs of indices with Riemann/Ricci tensors.
3. Leave some indices uncontracted (these become the free indices).
4. The resulting identity is a tensorial DDI.

The number of free indices left depends on how many contractions are
performed. Each tensorial DDI reduces the number of independent
tensorial monomials at the corresponding rank and degree.

### Dual Relations (Level 6)

The identity `*R*_{abcd} = R_{abcd}` in d=4 is already a tensorial
identity (rank 4). For tensorial monomials involving dual Riemann
tensors (via DualRInv), this identity applies directly. The existing
DualRInv infrastructure (dual_rinv.jl) already handles the
representation; TInvar just needs to apply the double-dual identity as
a reduction rule on TRInv elements that contain dualized factors.

## 7. Integration with Existing Pipeline

### Option A: Extend `riemann_simplify`

Extend the existing `riemann_simplify` function to accept expressions
with free indices. The current `is_riemann_monomial` check explicitly
rejects expressions with free indices (line 59 of `simplify_levels.jl`).
Removing this check and extending each level to handle TRInv would be the
most uniform approach.

### Option B: Separate `tensorial_simplify` Function

Create a separate top-level function `tensorial_simplify` that applies
the same 6-level algorithm but works on tensorial monomials. This avoids
modifying `riemann_simplify` and its test suite.

### Recommendation: Option B (Separate Function)

The scalar and tensorial cases have different performance characteristics
and different relation databases. Keeping them separate avoids the risk
of regressing the scalar pipeline (which has pinned benchmark term
counts). The common infrastructure (xperm canonicalization, Bianchi
identity application, DDI generation) can be shared as internal
functions.

### Proposed API

```julia
"""
    tensorial_simplify(expr::TensorExpr;
                        registry=current_registry(),
                        dim::Union{Int,Nothing}=nothing,
                        maxlevel::Int=6,
                        covd::Symbol=:D) -> TensorExpr

Simplify a tensorial Riemann monomial (curvature expression with free
indices) using the 6-level Invar algorithm adapted for the tensorial case.

Levels 1-6 are analogous to `riemann_simplify`:
1. Permutation symmetries (xperm with proper free/dummy separation)
2. Cyclic symmetry (first Bianchi identity)
3. Differential Bianchi identity
4. Derivative commutation
5. Dimensionally-dependent identities (tensorial DDIs)
6. Dual relations

Unlike `riemann_simplify`, this function handles expressions with free
indices and uses the full xperm double-coset algorithm (not BFS orbit
enumeration).
"""
function tensorial_simplify(expr::TensorExpr; ...)

"""
    tensorial_monomials(degree::Int, free_idxs::Vector{TIndex};
                         registry=current_registry(),
                         dim::Int=4) -> Vector{TensorExpr}

Enumerate all independent tensorial Riemann monomials of the given degree
with the specified free-index structure.

Returns a list of canonical TensorExpr's forming a basis for the vector
space of degree-k tensorial monomials with the given free indices.
"""
function tensorial_monomials(degree, free_idxs; ...)

"""
    trinv_basis(degree::Int, rank::Int;
                symmetry::Union{Nothing, YoungTableau}=nothing,
                dim::Int=4) -> Vector{TRInv}

Enumerate all independent TRInv canonical forms at given degree and rank,
optionally projected onto a specific symmetry type via Young tableaux.
"""
function trinv_basis(degree, rank; ...)
```

## 8. Estimated Complexity and Implementation Plan

### Phase 1: TRInv Core (Estimated: 3-4 issues)

1. **TRInv struct and constructors** (~100 LOC)
   - Type definition with validation
   - Conversion from/to RInv
   - Conversion from/to TensorExpr

2. **TRInv canonicalization via xperm** (~200 LOC)
   - `_trinv_slot_generators` with free-index-aware factor exchange
   - Name assignment with proper free/dummy separation
   - Call to `xperm_canonical_perm_ext`
   - Reconstruction of canonical TRInv

3. **Equality and hashing** (~30 LOC)
   - Canonical comparison
   - Hash based on canonical contraction + free_positions

### Phase 2: Enumeration (Estimated: 2-3 issues)

4. **`enumerate_tinvar_monomials`** (~150 LOC)
   - Build on `all_contractions` from ansatz.jl
   - Free-index-aware deduplication
   - Optional Young projection

5. **Ground-truth verification** (~200 LOC test)
   - Verify independent counts against known results
   - Degree 1: rank-2 (Ricci), rank-4 (Riemann)
   - Degree 2: rank-2 monomials in d=4

### Phase 3: Relations (Estimated: 3-4 issues)

6. **Tensorial Bianchi relations** (~100 LOC)
   - Extend `apply_bianchi_cyclic` for TRInv
   - Generate and collect linear relations

7. **Tensorial DDIs** (~200 LOC)
   - Extend `generate_ddi_rules` for tensorial identities
   - Partial contractions of the generalized delta

8. **`tensorial_simplify` orchestrator** (~100 LOC)
   - Wire up levels 1-6 for tensorial monomials
   - Integration tests

### Phase 4: Integration (Estimated: 1-2 issues)

9. **Wire into `simplify` pipeline** (~50 LOC)
   - Optional: route expressions with free curvature indices through
     tensorial simplification automatically

10. **Database extension** (~varies)
    - Extend the Invar database (design in invar_database_design.md) to
      include tensorial relations

### Total Estimated Effort

- New source code: ~900 LOC across ~6 files in `src/invariants/`
- New test code: ~400 LOC across 2-3 test files
- 10-12 implementation issues

### Risk Assessment

- **Low risk**: TRInv struct, canonicalization via xperm (well-understood
  algorithm, existing FFI infrastructure)
- **Medium risk**: Enumeration at high degree/rank (combinatorial
  explosion of perfect matchings; may need smarter pruning)
- **High risk**: Tensorial DDIs (the generalized-delta contraction with
  free indices is nontrivially more complex than the scalar case;
  requires careful dimension-dependent analysis)

## 9. References

1. Garcia-Parrado, A. & Martin-Garcia, J.M. (2007).
   "Spinor calculus on five-dimensional spacetimes."
   Comp. Phys. Comm. 176, 246--263. Section 4 (Invar algorithm).

2. Zakhary, E. & McIntosh, C.B.G. (1997).
   "A complete set of Riemann invariants."
   Gen. Rel. Grav. 29, 539--581.

3. Portugal, R. (1998).
   "An algorithm to simplify tensor expressions."
   Comp. Phys. Comm. 115, 87--106.

4. Manssur, L.R.U. & Portugal, R. (2001).
   "The Canon package: a fast kernel for tensor manipulators."
   Comp. Phys. Comm. 141, 273--295.

5. Fulling, S.A., King, R.C., Wybourne, B.G. & Cummins, C.J. (1992).
   "Normal forms for tensor polynomials: I. The Riemann tensor."
   Class. Quantum Grav. 9, 1151--1197.

6. Martin-Garcia, J.M. (2008). "xPerm: fast index canonicalization for
   tensor computer algebra." Comp. Phys. Comm. 179, 597--603.
