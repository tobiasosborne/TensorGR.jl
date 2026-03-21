# SymH Representation Design -- TGR-4zb.1

Design document for the SymH (Symmetry Handler) representation in TensorGR.jl.

## 1. Problem Statement

TensorGR's current symmetry infrastructure handles **monoterm symmetries** -- single
permutations with a sign. This covers:

- Symmetric pairs: `g_{ab} = g_{ba}` (swap two slots, sign +1)
- Antisymmetric pairs: `R_{abcd} = -R_{bacd}` (swap two slots, sign -1)
- Pair symmetry: `R_{abcd} = R_{cdab}` (swap two pairs, sign +1)
- Full symmetry/antisymmetry over n slots

These are represented as `SymmetrySpec` types (`Symmetric`, `AntiSymmetric`,
`PairSymmetric`, `RiemannSymmetry`, `FullySymmetric`, `FullyAntiSymmetric`)
and converted to permutation generators for xperm.c via `symmetry_generators()`.

The Invar pipeline and advanced tensor algebra require capabilities beyond
monoterm symmetries:

1. **Multi-term symmetries**: Relations involving sums of permuted terms. The
   first Bianchi identity `R_{a[bcd]} = 0` (equivalently
   `R_{abcd} + R_{acdb} + R_{adbc} = 0`) cannot be encoded as a single
   permutation generator. Currently handled via rewrite rules in `bianchi.jl`.

2. **Young symmetries**: Irreducible representations of S_n specified by Young
   tableaux. The Riemann tensor has the Young symmetry of the partition [2,2].
   Currently handled only at the expression level via `young_symmetrize()` and
   `young_project()` in `young.jl`, not integrated into the symmetry type system.

3. **Mixed symmetries**: When an imposed symmetry interacts with an internal
   (tensor product) symmetry, the combined symmetry group is a subgroup of
   S_n that is neither fully symmetric nor fully antisymmetric. xAct's
   SymManipulator computes this via `GeneralMixedSymmetry` using
   `SymmetryOfProductOfGroupsBruteForce`.

4. **Component counting**: Given a tensor's symmetry, compute the number of
   independent components in d dimensions. Currently not supported. Required
   for the Invar invariant enumeration pipeline.

5. **Hermitian symmetries**: Complex conjugation symmetries for spinor-valued
   tensors. Not currently needed but should be planned for.

## 2. Prior Art: xAct SymManipulator

xAct's SymManipulator (Thomas Backdahl, 2011-2026) provides the `SymH` type:

```mathematica
SymH[headlist, sym, label][indices]
```

where:
- `headlist` is the list of tensor heads whose product is being symmetrized
- `sym` is the imposed symmetry group (as `Symmetric`, `Antisymmetric`, or
  `StrongGenSet`)
- `label` is a display label

Key operations:
- `ImposeSym[expr, inds, sym]` -- impose a symmetry on an expression
- `ExpandSym[expr]` -- expand SymH objects into sums of permuted terms
- `SymmetryGroupOfTensor[SymH[...]]` -- compute the combined symmetry group
  of the product of tensors with the imposed symmetry
- `InternalCommutingSymmetry` -- the part of the internal symmetry that
  commutes with the imposed symmetry
- `IrrDecompose` -- irreducible decomposition over spinor indices

The symmetry group computation in SymManipulator dispatches on the type of
imposed symmetry:
- Symmetric case: `SymmetryOfProductOfGroupsSymmetricCase2b` (efficient)
- Antisymmetric case: `AntisymmetricMixedSymmetry` (via set stabilizer)
- General case: `GeneralMixedSymmetry` (brute force orbit enumeration)

The brute-force path computes the full orbit of the internal symmetry group
under the action of the imposed group, which is exponential in the number
of slots.

## 3. Proposed SymH Type Hierarchy

### 3.1 Core Types

```julia
"""
Abstract supertype for all symmetry specifications (new unified hierarchy).
"""
abstract type AbstractSymmetry end

"""
    MonotermSym

A monoterm symmetry: a single permutation with a sign (+1 or -1).
This subsumes the existing SymmetrySpec union types.

# Fields
- `perm::Vector{Int}` -- permutation in images notation on n slots (1-indexed)
- `sign::Int`         -- +1 (symmetric) or -1 (antisymmetric)
"""
struct MonotermSym <: AbstractSymmetry
    perm::Vector{Int}
    sign::Int    # +1 or -1
end

"""
    MultitermSym

A multi-term symmetry: a linear combination of permuted copies of a tensor
that equals zero.

    sum_i c_i * T_{sigma_i(a1 ... an)} = 0

Each term is (coefficient, permutation) where the permutation acts on slots.

# Fields
- `nslots::Int` -- number of index slots
- `terms::Vector{Tuple{Rational{Int}, Vector{Int}}}` -- (coefficient, permutation)

# Example
The first Bianchi identity R_{abcd} + R_{acdb} + R_{adbc} = 0 on slots 2,3,4:
```julia
MultitermSym(4, [
    (1//1, [1,2,3,4]),  # R_{abcd}
    (1//1, [1,3,4,2]),  # R_{acdb}
    (1//1, [1,4,2,3]),  # R_{adbc}
])
```
"""
struct MultitermSym <: AbstractSymmetry
    nslots::Int
    terms::Vector{Tuple{Rational{Int}, Vector{Int}}}
end

"""
    YoungSym

A symmetry specified by a Young tableau.

The Young tableau defines an irreducible representation of S_n. The
symmetry projector is the composition: antisymmetrize over columns,
then symmetrize over rows (the "Young symmetrizer" convention).

# Fields
- `tableau::YoungTableau` -- the Young tableau (rows of slot positions)
- `nslots::Int`           -- total number of index slots

# Notes
The Riemann tensor's index symmetry corresponds to the Young tableau
with shape [2,2] and slots [[1,3],[2,4]]:
```
  1  3
  2  4
```
This encodes: symmetrize over rows ({1,3} and {2,4}), then
antisymmetrize over columns ({1,2} and {3,4}).
"""
struct YoungSym <: AbstractSymmetry
    tableau::YoungTableau
    nslots::Int
end

"""
    HermitianSym

Hermitian symmetry for complex tensor fields. Relevant for spinors
where T_{AB'} = conj(T_{B'A}).

# Fields
- `slot_pairs::Vector{Tuple{Int,Int}}` -- pairs of slots related by
  complex conjugation
- `phase::Int` -- overall phase: +1 (Hermitian) or -1 (anti-Hermitian)
"""
struct HermitianSym <: AbstractSymmetry
    slot_pairs::Vector{Tuple{Int,Int}}
    phase::Int   # +1 or -1
end
```

### 3.2 Composite Symmetry

```julia
"""
    SymH

Unified symmetry handler for a tensor or tensor product.

Stores both the monoterm symmetry group (as generators for xperm) and
any additional multi-term / Young symmetries that constrain the tensor
beyond what xperm can canonicalize.

# Fields
- `nslots::Int`                            -- total number of index slots
- `generators::Vector{MonotermSym}`        -- monoterm symmetry generators
  (These are the permutation + sign generators passed to xperm.c.)
- `multiterm::Vector{MultitermSym}`        -- multi-term symmetry relations
  (Linear relations among permuted copies. Not used by xperm.)
- `young::Union{YoungSym, Nothing}`        -- Young tableau specification
  (Defines the irreducible representation, if applicable.)
- `hermitian::Union{HermitianSym, Nothing}` -- Hermitian symmetry
  (For complex tensor fields.)
- `dim::Union{Int, Nothing}`               -- manifold dimension
  (Required for dimension-dependent component counting.)

# Notes
The `generators` field is the bridge to xperm.c: it contains exactly
the data that `symmetry_generators()` currently produces. The remaining
fields capture symmetry information that xperm cannot use directly but
that is required for the Invar pipeline (component counting, multi-term
reduction, irreducible decomposition).
"""
struct SymH
    nslots::Int
    generators::Vector{MonotermSym}
    multiterm::Vector{MultitermSym}
    young::Union{YoungSym, Nothing}
    hermitian::Union{HermitianSym, Nothing}
    dim::Union{Int, Nothing}
end
```

### 3.3 Backward Compatibility

The existing `SymmetrySpec` types (`Symmetric`, `AntiSymmetric`, etc.) remain
unchanged. They are the registry-level specification. `SymH` is constructed
from them when advanced symmetry operations are needed:

```julia
"""
    SymH(specs::Vector{SymmetrySpec}, nslots::Int; dim=nothing) -> SymH

Construct a SymH from the existing SymmetrySpec registry format.
"""
function SymH(specs::Vector{SymmetrySpec}, nslots::Int; dim=nothing)
    # Convert SymmetrySpec to MonotermSym generators
    generators = MonotermSym[]
    for spec in specs
        for perm_obj in _sym_to_perm(spec, nslots + 2)
            pdata = [Int(perm_obj.data[i]) for i in 1:nslots]
            sign = perm_obj.data[nslots+1] == nslots+1 ? +1 : -1
            push!(generators, MonotermSym(pdata, sign))
        end
    end

    # Detect Riemann and add Bianchi multi-term symmetry
    multiterm = MultitermSym[]
    if any(s -> s isa RiemannSymmetry, specs)
        push!(multiterm, bianchi_multiterm())
    end

    # Detect Young symmetry from the SymmetrySpec
    young = _detect_young(specs, nslots)

    SymH(nslots, generators, multiterm, young, nothing, dim)
end
```

## 4. Monoterm vs Multi-term Symmetry Representation

### 4.1 Monoterm Symmetries (Status Quo)

Monoterm symmetries are single permutations acting on index slots with a sign
factor. They form a group G < S_n x Z_2 (the "signed permutation group" or
"hyperoctahedral" embedding). xperm.c handles these by encoding the sign in
two extra points (the "sign bits" at positions n+1, n+2).

Current representation: `_sym_to_perm(spec, n)` converts each `SymmetrySpec`
to one or more `Perm` objects on n = nslots + 2 points. The canonicalize
pipeline in `_canonicalize_product` passes these generators to
`xperm_canonical_perm`.

**No change needed.** The `MonotermSym` type in Section 3.1 is a thin
abstraction over the existing representation. The conversion
`MonotermSym -> Perm` is:

```julia
function to_xperm_perm(ms::MonotermSym, nslots::Int)
    n = nslots + 2
    p = collect(Int32, 1:n)
    for i in 1:nslots
        p[i] = Int32(ms.perm[i])
    end
    if ms.sign == -1
        p[n-1], p[n] = p[n], p[n-1]
    end
    Perm(p)
end
```

### 4.2 Multi-term Symmetries

Multi-term symmetries are algebraic relations of the form:

    sum_i c_i * T_{sigma_i(indices)} = 0

where `sigma_i` are permutations and `c_i` are rational coefficients.

**Key examples:**

1. **First Bianchi identity** (algebraic):
   `R_{abcd} + R_{acdb} + R_{adbc} = 0`
   This is `R_{a[bcd]} = 0` -- the cyclic sum over the last three indices.

2. **Second Bianchi identity** (differential):
   `nabla_a R_{bcde} + nabla_b R_{cade} + nabla_c R_{abde} = 0`

3. **Generalized Kronecker delta vanishing** (DDI):
   `delta^{a1...a_{d+1}}_{b1...b_{d+1}} = 0` in d dimensions, contracted
   with curvature tensors.

**Current handling:** Multi-term symmetries are implemented as `RewriteRule`
objects in `bianchi.jl` and `ddi_rules.jl`. The simplify pipeline applies
these rules in its fixed-point loop. This works but has limitations:

- Pattern matching is fragile (order-dependent, misses non-trivial contractions)
- Cannot be used for component counting (need the full symmetry space, not
  just rewrite rules)
- Not composable with monoterm symmetries in the canonicalization step

**Proposed representation:** `MultitermSym` stores the relation as a vector
of (coefficient, permutation) pairs. The relation acts on the index slots of
a single tensor (or tensor product). For multi-tensor relations (like the
Bianchi identity acting on a Riemann factor inside a product), the permutation
acts on the slots of that factor and the relation is applied factor-by-factor.

**Canonical representation of multi-term symmetries:**

A tensor T with monoterm group G and multi-term relations {R_i} lives in
the quotient space V^{tensor n} / (G + span(R_i)). The independent components
are the dimension of this quotient. To work with the quotient effectively,
we need the relations in row-echelon form (or reduced row-echelon form) when
viewed as a matrix over the permutation representation space.

```julia
"""
    multiterm_reduction_matrix(sym::SymH) -> Matrix{Rational{Int}}

Construct the reduction matrix for multi-term symmetries.

Rows correspond to multi-term relations. Columns correspond to
elements of the permutation representation space (orbits under
the monoterm group). The matrix is in row-echelon form.

The null space of this matrix (columns not in a pivot position)
gives the independent components.
"""
function multiterm_reduction_matrix(sym::SymH)
    # 1. Enumerate orbits of the monoterm group on S_n
    # 2. For each multi-term relation, express it as a linear combination
    #    of orbit representatives
    # 3. Row-reduce the resulting matrix
    ...
end
```

### 4.3 Interaction Between Monoterm and Multi-term Symmetries

The monoterm symmetry group G acts on the space of multi-term relations.
If R is a multi-term relation and g is in G, then g(R) is also a relation.
This means the multi-term relations should be stored modulo the action of G.

For canonicalization, the two types of symmetry interact:

1. **xperm canonicalization** uses monoterm symmetries to find the canonical
   representative of a product (Level 1 of Invar).

2. **Multi-term reduction** uses multi-term symmetries to express certain
   orbit representatives as linear combinations of others (Level 2 of Invar).

The Invar pipeline already applies these in sequence (Level 1 then Level 2).
The SymH type formalizes this by storing both together, enabling operations
like `independent_component_count` that need both.

## 5. Young Tableaux Integration

### 5.1 From Young Tableaux to Symmetry Groups

A Young tableau with shape lambda (a partition of n) defines:
- A **row symmetrizer** P: symmetrize over each row
- A **column antisymmetrizer** Q: antisymmetrize over each column
- The **Young symmetrizer** Y = Q * P (convention: first P, then Q)
- The **Young projector** e_lambda = (d_lambda / n!) * Y where d_lambda
  is the dimension of the irreducible representation

The symmetry of a tensor described by a Young tableau lambda is:
- **Monoterm part**: the generators of P and Q (symmetric transpositions
  for rows, antisymmetric transpositions for columns)
- **Multi-term part**: the relations that annihilate the orthogonal
  complement of the Young projector's image

For the Riemann tensor (shape [2,2]):
- Rows: {1,3} and {2,4} -- symmetric (pair symmetry)
- Columns: {1,2} and {3,4} -- antisymmetric
- The resulting group is not the full [2,2] irrep of S_4; it also has
  pair symmetry (R_{abcd} = R_{cdab}), which is a mixed symmetry.

### 5.2 Young Symmetry and xperm

xperm handles the monoterm part of the Young symmetry directly: the
row symmetrizers and column antisymmetrizers are just `Symmetric` and
`AntiSymmetric` generators. The multi-term relations (e.g., Bianchi
identity for Riemann) are the additional constraints that reduce the
number of independent components below what monoterm symmetries alone
would give.

**Important:** The Young projector is NOT a group element. It is a
linear operator on the tensor space. xperm canonicalizes within orbits
of the monoterm group; the Young projector projects onto a subspace
of those orbits. These are different operations.

### 5.3 Constructing SymH from Young Tableaux

```julia
"""
    SymH(yt::YoungTableau; dim=nothing) -> SymH

Construct a SymH from a Young tableau.

The monoterm generators are:
- Adjacent transpositions within each row (sign +1, symmetric)
- Adjacent transpositions within each column (sign -1, antisymmetric)

The multi-term relations encode the annihilation conditions: any tensor
component that is projected to zero by the Young projector satisfies a
linear relation among its permuted copies.
"""
function SymH(yt::YoungTableau; dim=nothing)
    nslots = sum(length(r) for r in yt.rows)
    generators = MonotermSym[]

    # Row symmetrizers (symmetric transpositions)
    for row in yt.rows
        for i in 1:length(row)-1
            p = collect(1:nslots)
            p[row[i]], p[row[i+1]] = p[row[i+1]], p[row[i]]
            push!(generators, MonotermSym(p, +1))
        end
    end

    # Column antisymmetrizers (antisymmetric transpositions)
    for col in young_columns(yt)
        for i in 1:length(col)-1
            p = collect(1:nslots)
            p[col[i]], p[col[i+1]] = p[col[i+1]], p[col[i]]
            push!(generators, MonotermSym(p, -1))
        end
    end

    young = YoungSym(yt, nslots)
    SymH(nslots, generators, MultitermSym[], young, nothing, dim)
end
```

### 5.4 Riemann's Young Symmetry

The Riemann tensor's full symmetry is:

1. **Monoterm** (handled by xperm via `RiemannSymmetry`):
   - `AntiSymmetric(1,2)`: R_{abcd} = -R_{bacd}
   - `AntiSymmetric(3,4)`: R_{abcd} = -R_{abdc}
   - `PairSymmetric(1,2,3,4)`: R_{abcd} = R_{cdab}

2. **Multi-term** (first Bianchi, NOT handled by xperm):
   - `R_{abcd} + R_{acdb} + R_{adbc} = 0`

3. **Young tableau**: shape [2,2], which encodes both (1) and (2).

The SymH for Riemann would be:

```julia
function riemann_symh(; dim=nothing)
    # Monoterm generators (same as RiemannSymmetry)
    gens = [
        MonotermSym([2,1,3,4], -1),   # anti(1,2)
        MonotermSym([1,2,4,3], -1),   # anti(3,4)
        MonotermSym([3,4,1,2], +1),   # pair(1,2,3,4)
    ]

    # Multi-term: first Bianchi identity
    bianchi = MultitermSym(4, [
        (1//1, [1,2,3,4]),  # R_{abcd}
        (1//1, [1,3,4,2]),  # R_{acdb}
        (1//1, [1,4,2,3]),  # R_{adbc}
    ])

    # Young tableau: [2,2]
    young = YoungSym(YoungTableau([[1,3],[2,4]]), 4)

    SymH(4, gens, [bianchi], young, nothing, dim)
end
```

## 6. Component Counting Algorithm

### 6.1 Problem

Given a tensor T with n index slots on a manifold of dimension d, and a
SymH specifying its symmetries, compute the number of algebraically
independent components.

**Examples (in d=4):**
- Metric g_{ab}: symmetric, 2 slots -> 4*5/2 = 10 components
- Riemann R_{abcd}: RiemannSymmetry + Bianchi -> 20 components
- Weyl C_{abcd}: trace-free Riemann -> 10 components
- Ricci R_{ab}: symmetric -> 10 components

### 6.2 Algorithm

The number of independent components is computed in three stages:

**Stage 1: Monoterm symmetries (group orbit counting)**

The monoterm symmetry group G acts on the set of index tuples
{1,...,d}^n. The number of orbits equals the number of independent
components under monoterm symmetries alone. By Burnside's lemma:

    N_monoterm = (1/|G|) * sum_{g in G} |Fix(g)|

where Fix(g) is the number of index tuples fixed by g. For a
permutation g with cycle structure (c_1, c_2, ..., c_k), the number
of fixed tuples is d^k (each cycle must be assigned a constant value).

For signed permutations (sign = -1), a fixed tuple must satisfy
T_{g(i)} = -T_{i}, which means T = 0 for any tuple in the support
of g. Hence Fix(g) = d^{k_fixed} where k_fixed counts only cycles
whose associated sign is +1.

```julia
"""
    monoterm_component_count(sym::SymH) -> Int

Count independent components under monoterm symmetries only,
using Burnside's lemma.
"""
function monoterm_component_count(sym::SymH)
    d = sym.dim
    d === nothing && error("dimension required for component counting")

    G = _enumerate_group(sym.generators, sym.nslots)
    total = 0
    for (perm, sign) in G
        # Count cycles in perm
        visited = falses(sym.nslots)
        n_positive_cycles = 0
        for i in 1:sym.nslots
            visited[i] && continue
            cycle_len = 0
            cycle_sign = +1
            j = i
            while !visited[j]
                visited[j] = true
                cycle_len += 1
                j = perm[j]
            end
            # For antisymmetric generators, odd-length cycles in the
            # support contribute sign = -1
            if sign == -1
                # This is a simplification; the proper handling requires
                # tracking the sign through group composition
            end
            n_positive_cycles += 1
        end
        total += d^n_positive_cycles
    end
    total ÷ length(G)
end
```

**Proper implementation note:** The above sketch uses `_enumerate_group` which
constructs the full group from generators. For small groups (Riemann has |G|=8,
RiemannSymmetry group), this is fast. For larger groups, use the Schreier-Sims
algorithm (already available via `xperm_schreier_sims`) to compute |G| and
cycle indices without enumerating all elements.

The exact formula using the cycle index polynomial Z(G; x_1, ..., x_n)
evaluated at x_i = d (for unsigned cycles) or x_i = d for even-length
cycles and x_i = 0 for odd-length cycles (for signed permutations) gives
the count directly. This avoids enumerating group elements.

**Stage 2: Multi-term symmetries (linear algebra)**

Multi-term relations reduce the count further. Each relation
`sum_i c_i T_{sigma_i(a1...an)} = 0` imposes one linear constraint
on the d^n components (or on the monoterm-reduced components).

The number of independent constraints is the rank of the matrix formed
by all multi-term relations expressed in the monoterm-orbit basis.

```julia
"""
    multiterm_reduction_count(sym::SymH) -> Int

Count the number of independent constraints from multi-term symmetries.
"""
function multiterm_reduction_count(sym::SymH)
    isempty(sym.multiterm) && return 0
    M = multiterm_reduction_matrix(sym)
    # rank of M = number of dependent components
    _rational_rank(M)
end
```

**Stage 3: Final count**

```julia
"""
    independent_component_count(sym::SymH) -> Int

Count the number of algebraically independent components.
"""
function independent_component_count(sym::SymH)
    n_mono = monoterm_component_count(sym)
    n_multi = multiterm_reduction_count(sym)
    n_mono - n_multi
end
```

### 6.3 Verification

The component counts for standard GR tensors serve as ground truth:

| Tensor          | d  | Monoterm | Multi-term | Independent |
|-----------------|----|----------|------------|-------------|
| Metric g_{ab}   | d  | d(d+1)/2 | 0          | d(d+1)/2    |
| Riemann R_{abcd}| d  | d^2(d^2-1)/12 + d(d-1)/2 | see below | d^2(d^2-1)/12 |
| Riemann R_{abcd}| 4  | 21       | 1          | 20          |
| Ricci R_{ab}    | d  | d(d+1)/2 | 0          | d(d+1)/2    |
| Ricci R_{ab}    | 4  | 10       | 0          | 10          |
| Weyl C_{abcd}   | 4  | 20       | 10 (trace) | 10          |

The Riemann tensor in d dimensions has d^2(d^2-1)/12 independent components.
In d=4, that is 20. The monoterm symmetries (antisymmetry + pair symmetry)
give 21 orbits; the single Bianchi constraint reduces this to 20.

For the Weyl tensor, the additional trace-free constraints (Weyl is the
trace-free part of Riemann) remove 10 components in d=4, leaving 10.

## 7. Interface with xperm and Existing canonicalize()

### 7.1 Current Pipeline

```
simplify pipeline:
  expand_products
  -> contract_metrics
  -> contract_curvature
  -> canonicalize  (calls _canonicalize_product -> xperm.c)
  -> [commute_covds]
  -> collect_terms
  -> apply_rules   (Bianchi rules, DDI rules, etc.)
```

The `canonicalize` step uses xperm with monoterm generators only. Multi-term
symmetries are handled post-hoc via `apply_rules`.

### 7.2 Proposed Enhanced Pipeline

With SymH, the pipeline gains a pre-reduction step:

```
simplify pipeline (enhanced):
  expand_products
  -> contract_metrics
  -> contract_curvature
  -> canonicalize_monoterm  (xperm.c, unchanged)
  -> [reduce_multiterm]     (NEW: apply multi-term reductions)
  -> [commute_covds]
  -> collect_terms
  -> apply_rules
```

The `reduce_multiterm` step uses the `MultitermSym` relations from the SymH
to rewrite expressions. For each factor in a product that has multi-term
symmetries, the step checks if the factor's canonical form (from xperm) can
be further reduced using the multi-term relations.

**Critical constraint:** This step must NOT interfere with the existing
canonicalize behavior. The xperm canonicalization MUST remain unchanged
(see CLAUDE.md: "Do NOT sort TSum terms or batch-rename in
`_normalize_dummies`"). The multi-term reduction is additive: it only
replaces terms that are in the dependent set (those that can be expressed
as linear combinations of independent terms via multi-term relations).

### 7.3 xperm Compatibility

The `SymH.generators` field converts directly to xperm generators:

```julia
function to_xperm_generators(sym::SymH)
    Perm[to_xperm_perm(g, sym.nslots) for g in sym.generators]
end
```

This is a drop-in replacement for the current `symmetry_generators(specs, nslots)`
call in `_canonicalize_product`. The migration path is:

1. Store `SymH` alongside (or instead of) `Vector{SymmetrySpec}` in
   `TensorProperties`
2. In `_canonicalize_product`, use `to_xperm_generators(symh)` instead
   of `symmetry_generators(props.symmetries, nslots)`
3. Optionally, after xperm canonicalization, apply `reduce_multiterm`
   if the SymH has non-empty `multiterm` relations

### 7.4 Integration with the Invar Pipeline

The Invar 6-level pipeline (`riemann_simplify`) already handles multi-term
symmetries at each level. SymH provides a unified entry point:

- **Level 1** (permutation symmetries): `sym.generators` -> xperm
- **Level 2** (cyclic/Bianchi): `sym.multiterm` (first Bianchi)
- **Level 5** (DDIs): Dimension-dependent `MultitermSym` relations generated
  from `sym.dim` and the DDI infrastructure in `ddi_rules.jl`
- **Level 6** (dual relations): DualRInv infrastructure already handles this

The `RInv` type in `rinv.jl` uses its own symmetry group construction
(`_rinv_slot_generators`, `rinv_symmetry_group`). These should be unified
to construct from SymH:

```julia
function rinv_symh(degree::Int; dim=nothing)
    nslots = 4 * degree
    generators = MonotermSym[]

    # Per-factor Riemann monoterm symmetries
    for f in 1:degree
        off = 4(f-1)
        push!(generators, MonotermSym(_anti_perm(off+1, off+2, nslots), -1))
        push!(generators, MonotermSym(_anti_perm(off+3, off+4, nslots), -1))
        push!(generators, MonotermSym(_pair_perm(off+1, off+2, off+3, off+4, nslots), +1))
    end

    # Inter-factor transpositions
    for f in 1:degree-1
        push!(generators, MonotermSym(_block_swap_perm(f, f+1, 4, nslots), +1))
    end

    # Multi-term: Bianchi for each factor
    multiterm = MultitermSym[]
    for f in 1:degree
        off = 4(f-1)
        push!(multiterm, bianchi_multiterm_at(off, nslots))
    end

    SymH(nslots, generators, multiterm, nothing, nothing, dim)
end
```

## 8. Public API Proposal

### 8.1 Construction

```julia
# From existing SymmetrySpec (backward compatible)
SymH(specs::Vector{SymmetrySpec}, nslots::Int; dim=nothing) -> SymH

# From Young tableau
SymH(yt::YoungTableau; dim=nothing) -> SymH

# Built-in for common tensors
riemann_symh(; dim=nothing) -> SymH
weyl_symh(; dim=4) -> SymH     # Riemann + trace-free constraints
ricci_symh(; dim=nothing) -> SymH

# Add multi-term relation to existing SymH
add_multiterm(sym::SymH, relation::MultitermSym) -> SymH
```

### 8.2 Queries

```julia
# Component counting
independent_component_count(sym::SymH) -> Int
monoterm_component_count(sym::SymH) -> Int
multiterm_reduction_count(sym::SymH) -> Int

# Symmetry group properties
group_order(sym::SymH) -> Int
is_subgroup(sym1::SymH, sym2::SymH) -> Bool

# Check if expression respects symmetry
has_symmetry(expr::TensorExpr, sym::SymH) -> Bool
```

### 8.3 Operations

```julia
# Apply symmetry to expression
impose_symmetry(expr::TensorExpr, sym::SymH, idxs::Vector{Symbol}) -> TensorExpr
expand_symmetry(expr::TensorExpr, sym::SymH) -> TensorExpr

# Reduce using multi-term relations
reduce_multiterm(expr::TensorExpr, sym::SymH) -> TensorExpr

# Young projector
young_project(expr::TensorExpr, sym::SymH, idxs::Vector{Symbol}) -> TensorExpr

# Irreducible decomposition
irreducible_decompose(expr::TensorExpr, sym::SymH) -> Vector{TensorExpr}
```

### 8.4 xperm Interface

```julia
# Convert to xperm generators
to_xperm_generators(sym::SymH) -> Vector{Perm}

# Canonical form under monoterm symmetries
canonicalize_monoterm(expr::TensorExpr, sym::SymH) -> TensorExpr
```

## 9. Implementation Plan

### Phase 1: Core Types (Estimated: 2-3 issues)

1. **Define `AbstractSymmetry` hierarchy**: `MonotermSym`, `MultitermSym`,
   `YoungSym`, `HermitianSym`, `SymH` in a new file `src/symmetry/symh.jl`.
   No changes to existing code.

2. **Conversion from `SymmetrySpec`**: `SymH(specs, nslots)` constructor that
   converts the existing `SymmetrySpec` vector to a `SymH`. Verify round-trip:
   `to_xperm_generators(SymH(specs, n)) == symmetry_generators(specs, n)`.

3. **Tests**: Verify that `SymH` round-trips for all existing tensor
   registrations (metric, Riemann, Ricci, Weyl, etc.).

### Phase 2: Multi-term Symmetries (Estimated: 3-4 issues)

4. **`MultitermSym` for Bianchi**: Encode the first Bianchi identity as a
   `MultitermSym`. Verify that applying it to a Riemann expression produces
   the correct zero result.

5. **Multi-term reduction engine**: Implement `reduce_multiterm` that applies
   multi-term relations to simplify expressions. This replaces the current
   `apply_bianchi_cyclic` in `simplify_levels.jl` with a general mechanism.

6. **Integration with simplify pipeline**: Add optional `reduce_multiterm`
   step between `canonicalize` and `collect_terms`. Gate behind a flag
   initially (`use_symh=false`) for safety.

7. **Tests**: Verify that Level 2 Invar results are unchanged when using
   the new `reduce_multiterm` vs the old `apply_bianchi_cyclic`.

### Phase 3: Component Counting (Estimated: 2-3 issues)

8. **Burnside's lemma implementation**: Compute monoterm component counts
   using the cycle index polynomial. Verify against known values (metric: 10,
   Riemann: 21 monoterm, 20 total in d=4).

9. **Multi-term constraint counting**: Linear algebra over Q to compute the
   rank of the multi-term reduction matrix. Verify: Riemann in d=4 has 1
   independent multi-term constraint, giving 21 - 1 = 20.

10. **API**: `independent_component_count(sym)` public function.

### Phase 4: Young Integration (Estimated: 2-3 issues)

11. **`YoungSym` construction**: Build SymH from Young tableau. Verify that
    the generated monoterm generators match `symmetry_generators` for known
    cases (Riemann = [2,2]).

12. **Irreducible decomposition**: Implement `irreducible_decompose` using
    the Young projector. This generalizes the existing `young_project` in
    `young.jl`.

13. **Integration with Invar**: Use SymH's Young information in the invariant
    enumeration pipeline to classify invariants by their Young symmetry type.

### Phase 5: Integration and Optimization (Estimated: 2-3 issues)

14. **Store SymH in TensorProperties**: Add an optional `symh` field to
    `TensorProperties`. Lazily constructed from `specs` on first access.

15. **Optimize RInv**: Unify `rinv_symmetry_group` with SymH construction.
    Use SymH's component counting for the invariant enumeration pipeline.

16. **Benchmark verification**: Ensure that all 12 benchmarks (152 tests)
    pass with unchanged term counts. This is the critical regression gate.

### Complexity Estimates

| Component               | Lines (est.) | Difficulty |
|--------------------------|-------------|------------|
| Type definitions         | ~100        | Low        |
| SymmetrySpec conversion  | ~80         | Low        |
| MultitermSym engine      | ~200        | Medium     |
| Component counting       | ~150        | Medium     |
| Young integration        | ~120        | Medium     |
| Pipeline integration     | ~100        | High (regression risk) |
| Tests                    | ~400        | Medium     |
| **Total**                | **~1150**   |            |

## 10. Risks and Mitigations

### 10.1 Regression Risk

The simplify pipeline is highly tuned. Any change to `canonicalize()` or
term collection risks breaking benchmark term counts.

**Mitigation:** SymH is additive -- it does not modify the existing
`canonicalize()` code. The `reduce_multiterm` step is a new, optional step
that runs after xperm canonicalization. It can be gated behind a flag and
enabled only for the Invar pipeline initially.

### 10.2 Performance

Enumerating the full monoterm group for Burnside's lemma is O(|G|). For
Riemann products of degree k, |G| = 8^k * k! (Riemann factor symmetry
times inter-factor permutations). At degree 5, |G| = 8^5 * 120 = 3,932,160.
This is feasible for one-time component counting but not for per-expression
canonicalization.

**Mitigation:** Use the Schreier-Sims algorithm (available via xperm.c) to
compute cycle indices and group orders without enumerating all elements. For
component counting, compute the cycle index polynomial symbolically.

### 10.3 Multi-term Reduction Completeness

The multi-term reduction engine must be complete: every dependent expression
must be rewritten to the independent basis. Incomplete reduction would
silently produce wrong invariant counts.

**Mitigation:** Verify completeness against the known Invar database counts
from Garcia-Parrado & Martin-Garcia (2007) Table 1. For degree 2 in d=4:
3 independent algebraic invariants (R^2, Ric^2, Riem^2 with Riem^2 =
4 Ric^2 - R^2 via Gauss-Bonnet).

### 10.4 xperm All-Free Mode

The current `_canonicalize_product` uses xperm in "all-free mode" (all
indices treated as free, dummies normalized separately). This is a
deliberate design choice documented in CLAUDE.md. SymH must preserve
this convention.

**Mitigation:** `to_xperm_generators(sym)` produces generators in the
same format as the existing `symmetry_generators()`. The only difference
is the source of the generators (SymH fields vs SymmetrySpec dispatch).
The xperm calling convention is unchanged.

## 11. References

1. Garcia-Parrado, A. & Martin-Garcia, J.M. (2007). "Spinor calculus on
   five-dimensional spacetimes." Comp. Phys. Comm. **176**, 246-253.
   -- Invar 6-level algorithm, component counting, canonicalization.

2. Manssur, L.R.U., Portugal, R. & Svaiter, B.F. (2004). "Group-theoretic
   approach for symbolic tensor manipulation: II. Dummy indices."
   Int. J. Mod. Phys. C **15**, 471-487.
   -- Multi-term symmetries, Young tableau integration with canonicalization.

3. Backdahl, T. (2011-2026). "SymManipulator: Symmetrized tensor expressions
   and irreducible decomposition." xAct package.
   -- SymH type, ImposeSym, mixed symmetry computation, IrrDecompose.

4. Butler, G. & Portugal, R. (2003). "Algorithmic simplification of tensor
   expressions." J. Symb. Comp. **36**, 159-188.
   -- xperm algorithm for monoterm canonicalization.

5. Fulling, S.A., King, R.C., Wybourne, B.G. & Cummins, C.J. (1992).
   "Normal forms for tensor polynomials. I: The Riemann tensor."
   Class. Quantum Grav. **9**, 1151-1197.
   -- Complete classification of Riemann invariants, cubic DDIs.

6. Zakhary, E. & McIntosh, C.B.G. (1997). "A Complete Set of Riemann
   Invariants." Gen. Rel. Grav. **29**, 539-581.
   -- Independent invariant sets, component counting.

7. Lovelock, D. (1971). "The Einstein tensor and its generalizations."
   J. Math. Phys. **12**, 498-501.
   -- Dimensionally-dependent identities from generalized Kronecker delta.
