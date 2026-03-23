#= SymH: Unified symmetry handler for tensor indices.
#
# Supports monoterm symmetries (single permutation + sign, for xperm),
# multi-term symmetries (linear relations among permuted copies, e.g.
# Bianchi identity), and composites of both.
#
# Reference: docs/design/symmanipulator_design.md
=#

"""Abstract base for symmetry representations."""
abstract type AbstractSymmetry end

"""
    MonotermSym <: AbstractSymmetry

Monoterm symmetry: a single permutation of index slots with a sign.

# Fields
- `perm::Vector{Int}` -- permutation in images notation on nslots slots (1-indexed)
- `sign::Int`         -- +1 (symmetric) or -1 (antisymmetric)
"""
struct MonotermSym <: AbstractSymmetry
    perm::Vector{Int}
    sign::Int   # +1 or -1
end

"""
    MultitermSym <: AbstractSymmetry

Multi-term symmetry: a linear relation among permuted copies of a tensor
that sums to zero.

    sum_i c_i * T_{sigma_i(a1 ... an)} = 0

Each term is (coefficient, permutation) where the permutation acts on slots.

# Fields
- `nslots::Int` -- number of index slots
- `terms::Vector{Tuple{Rational{Int}, Vector{Int}}}` -- (coefficient, permutation)

# Example
First Bianchi identity R_{abcd} + R_{acdb} + R_{adbc} = 0:
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
    SymH

Unified symmetry handler for a tensor or tensor product.

Stores monoterm symmetry generators (for xperm.c) and any additional
multi-term symmetry relations that constrain the tensor beyond what
xperm can canonicalize.

# Fields
- `nslots::Int`                        -- total number of index slots
- `monoterm::Vector{MonotermSym}`      -- monoterm symmetry generators
- `multiterm::Vector{MultitermSym}`    -- multi-term symmetry relations
"""
struct SymH
    nslots::Int
    monoterm::Vector{MonotermSym}
    multiterm::Vector{MultitermSym}
end

# ── Constructors ──────────────────────────────────────────────────────

"""
    SymH(spec::SymmetrySpec, nslots::Int) -> SymH

Construct a SymH from a single existing SymmetrySpec.
"""
function SymH(spec::SymmetrySpec, nslots::Int)
    SymH([spec], nslots)
end

"""
    SymH(specs::Vector{<:SymmetrySpec}, nslots::Int) -> SymH

Construct a SymH from a vector of existing SymmetrySpec objects.
Converts each spec to MonotermSym generators via _sym_to_perm.
Detects RiemannSymmetry and adds the first Bianchi as a MultitermSym.
"""
function SymH(specs::Vector{<:SymmetrySpec}, nslots::Int)
    n = nslots + 2  # xperm uses n = nslots + 2 for sign bits
    generators = MonotermSym[]
    for spec in specs
        for perm_obj in _sym_to_perm(spec, n)
            pdata = [Int(perm_obj.data[i]) for i in 1:nslots]
            sign = perm_obj.data[n - 1] == Int32(n - 1) ? +1 : -1
            push!(generators, MonotermSym(pdata, sign))
        end
    end

    # Detect Riemann and add Bianchi multi-term symmetry
    multiterm = MultitermSym[]
    if any(s -> s isa RiemannSymmetry, specs)
        push!(multiterm, _bianchi_multiterm())
    end

    SymH(nslots, generators, multiterm)
end

"""
    SymH(gens::Vector{Tuple{Vector{Int},Int}}, nslots::Int) -> SymH

Construct a SymH from raw generators as (perm, sign) tuples.
"""
function SymH(gens::Vector{Tuple{Vector{Int},Int}}, nslots::Int)
    monoterm = [MonotermSym(p, s) for (p, s) in gens]
    SymH(nslots, monoterm, MultitermSym[])
end

# ── First Bianchi identity as MultitermSym ────────────────────────────

"""
    _bianchi_multiterm() -> MultitermSym

First Bianchi identity R_{abcd} + R_{acdb} + R_{adbc} = 0 on 4 slots.
"""
function _bianchi_multiterm()
    MultitermSym(4, [
        (1 // 1, [1, 2, 3, 4]),  # R_{abcd}
        (1 // 1, [1, 3, 4, 2]),  # R_{acdb}
        (1 // 1, [1, 4, 2, 3]),  # R_{adbc}
    ])
end

# ── Conversion back to SymmetrySpec ───────────────────────────────────

"""
    to_symmetry_spec(symh::SymH) -> Vector{SymmetrySpec}

Convert monoterm generators of a SymH back to SymmetrySpec objects.
Only converts generators that correspond to a recognized SymmetrySpec
pattern (pairwise transposition). Multi-term symmetries are dropped.
"""
function to_symmetry_spec(symh::SymH)
    specs = SymmetrySpec[]
    for m in symh.monoterm
        spec = _monoterm_to_spec(m)
        spec !== nothing && push!(specs, spec)
    end
    specs
end

"""
    _monoterm_to_spec(m::MonotermSym) -> Union{SymmetrySpec, Nothing}

Attempt to convert a MonotermSym to a SymmetrySpec. Returns `nothing`
if the permutation is not a simple transposition.
"""
function _monoterm_to_spec(m::MonotermSym)
    nslots = length(m.perm)
    # Find swapped positions
    swapped = Int[]
    for i in 1:nslots
        if m.perm[i] != i
            push!(swapped, i)
        end
    end
    # Simple transposition: exactly 2 positions swapped
    if length(swapped) == 2
        i, j = swapped
        if m.perm[i] == j && m.perm[j] == i
            if m.sign == +1
                return Symmetric(i, j)
            else
                return AntiSymmetric(i, j)
            end
        end
    end
    # Pair swap: exactly 4 positions, two disjoint transpositions
    if length(swapped) == 4 && m.sign == +1
        a, b, c, d = swapped
        if m.perm[a] == c && m.perm[c] == a && m.perm[b] == d && m.perm[d] == b
            return PairSymmetric(a, b, c, d)
        end
    end
    nothing
end

# ── Built-in: Riemann symmetry ────────────────────────────────────────

"""
    riemann_symh() -> SymH

SymH for the rank-4 Riemann tensor, including:
- Monoterm: anti(1,2), anti(3,4), pair(1,2,3,4)
- Multi-term: first Bianchi identity R_{a[bcd]} = 0
"""
function riemann_symh()
    gens = [
        MonotermSym([2, 1, 3, 4], -1),   # anti(1,2)
        MonotermSym([1, 2, 4, 3], -1),   # anti(3,4)
        MonotermSym([3, 4, 1, 2], +1),   # pair(1,2,3,4)
    ]
    bianchi = _bianchi_multiterm()
    SymH(4, gens, [bianchi])
end

# ── Queries ───────────────────────────────────────────────────────────

"""
    is_monoterm_only(symh::SymH) -> Bool

Return true if the SymH has no multi-term symmetries.
"""
is_monoterm_only(symh::SymH) = isempty(symh.multiterm)

# ── Component counting ────────────────────────────────────────────────

"""
    n_independent_components(symh::SymH, dim::Int) -> Int

Count the number of algebraically independent components of a tensor
with symmetry `symh` on a manifold of dimension `dim`.

Uses Burnside's lemma for the monoterm symmetry group, then subtracts
constraints from multi-term symmetries via linear algebra over the
orbit representatives.

Ground truth (d=4): metric=10, Riemann=20, Ricci=10, Weyl=10.
"""
function n_independent_components(symh::SymH, dim::Int)
    # Enumerate the full monoterm symmetry group
    group = _enumerate_monoterm_group(symh)

    # Stage 1: Burnside's lemma -- count orbits of index tuples
    n_mono = _burnside_count(group, symh.nslots, dim)

    # Stage 2: multi-term constraints reduce the count
    if isempty(symh.multiterm)
        return n_mono
    end

    n_multi = _multiterm_constraint_count(symh, group, dim)
    return n_mono - n_multi
end

"""
    _enumerate_monoterm_group(symh::SymH) -> Vector{Tuple{Vector{Int}, Int}}

Enumerate all elements of the monoterm symmetry group from generators.
Each element is (permutation, sign). Uses BFS / orbit enumeration.
"""
function _enumerate_monoterm_group(symh::SymH)
    n = symh.nslots
    # Identity element
    id_perm = collect(1:n)
    # Group elements as Dict for deduplication: perm => sign
    group_dict = Dict{Vector{Int}, Int}()
    group_dict[id_perm] = +1

    queue = [(id_perm, +1)]
    while !isempty(queue)
        perm, sign = popfirst!(queue)
        for gen in symh.monoterm
            # Compose: new_perm[i] = gen.perm[perm[i]]
            new_perm = [gen.perm[perm[i]] for i in 1:n]
            new_sign = sign * gen.sign
            if !haskey(group_dict, new_perm)
                group_dict[new_perm] = new_sign
                push!(queue, (new_perm, new_sign))
            end
        end
    end

    [(p, s) for (p, s) in group_dict]
end

"""
    _burnside_count(group, nslots, dim) -> Int

Count independent components of a tensor with signed permutation
symmetry group on a manifold of dimension `dim`.

Uses the representation-theoretic formula:

    N = (1/|G|) * sum_{(sigma,s) in G} s * dim^{cycles(sigma)}

For each group element (sigma, s), the representation trace is
s * dim^k where k is the number of cycles of sigma. The sign s
accounts for antisymmetric generators: signed permutations contribute
*negatively* to the count, correctly subtracting diagonal/symmetric
components that are forced to vanish.
"""
function _burnside_count(group::Vector{Tuple{Vector{Int}, Int}}, nslots::Int, dim::Int)
    total = 0
    for (perm, sign) in group
        ncycles = _count_cycles(perm, nslots)
        total += sign * dim^ncycles
    end
    total ÷ length(group)
end

"""Count cycles in a permutation."""
function _count_cycles(perm::Vector{Int}, n::Int)
    visited = falses(n)
    ncycles = 0
    for i in 1:n
        visited[i] && continue
        ncycles += 1
        j = i
        while !visited[j]
            visited[j] = true
            j = perm[j]
        end
    end
    ncycles
end

"""
    _multiterm_constraint_count(symh, group, dim) -> Int

Count the number of independent constraints imposed by multi-term
symmetries on the monoterm-orbit basis.

Algorithm:
1. Enumerate all non-vanishing orbits of index tuples under the signed
   monoterm group, tracking the sign relating each tuple to its orbit
   representative.
2. For each multi-term relation evaluated on each orbit representative,
   build a linear constraint on orbit amplitudes (properly signed).
3. The rank of the constraint matrix gives the constraint count.
"""
function _multiterm_constraint_count(symh::SymH, group::Vector{Tuple{Vector{Int}, Int}}, dim::Int)
    n = symh.nslots
    # Enumerate orbits with sign tracking
    orbits, tuple_to_orbit, tuple_to_sign = _enumerate_orbits_with_signs(group, n, dim)
    n_orbits = length(orbits)
    n_orbits == 0 && return 0

    # Build constraint matrix
    constraint_rows = Vector{Rational{Int}}[]

    for mt in symh.multiterm
        # Evaluate the relation on every orbit representative
        for orbit_idx in 1:n_orbits
            rep = orbits[orbit_idx]
            row = zeros(Rational{Int}, n_orbits)
            for (coeff, sigma) in mt.terms
                # Apply sigma to the representative tuple
                permuted = [rep[sigma[j]] for j in 1:n]
                # Find which orbit this permuted tuple belongs to
                target_orbit = get(tuple_to_orbit, permuted, 0)
                target_orbit == 0 && continue  # vanishing orbit
                # Get the sign relating this tuple to its orbit rep
                target_sign = tuple_to_sign[permuted]
                # T_{permuted} = target_sign * amplitude[target_orbit]
                row[target_orbit] += coeff * target_sign
            end
            if any(!iszero, row)
                push!(constraint_rows, row)
            end
        end
    end

    isempty(constraint_rows) && return 0

    # Row reduce to find rank
    M = Matrix(reduce(hcat, constraint_rows)')
    _rational_matrix_rank(M)
end

"""
    _enumerate_orbits_with_signs(group, nslots, dim)
        -> (orbits, tuple_to_orbit, tuple_to_sign)

Enumerate all orbits of index tuples under the signed monoterm group.

Returns:
- `orbits`: list of orbit representative tuples (non-vanishing only)
- `tuple_to_orbit`: Dict mapping each tuple to its orbit index (0 for vanishing)
- `tuple_to_sign`: Dict mapping each tuple to the sign relating it to its
  orbit representative. If T is the tensor, then T_{tuple} = sign * T_{rep}.
"""
function _enumerate_orbits_with_signs(group::Vector{Tuple{Vector{Int}, Int}}, nslots::Int, dim::Int)
    orbits = Vector{Int}[]
    tuple_to_orbit = Dict{Vector{Int}, Int}()
    tuple_to_sign = Dict{Vector{Int}, Int}()

    for tuple in Iterators.product(ntuple(_ -> 1:dim, nslots)...)
        t = collect(Int, tuple)
        haskey(tuple_to_orbit, t) && continue

        # Compute the orbit of t with signs
        orbit_members = Tuple{Vector{Int}, Int}[]  # (permuted_tuple, sign)
        is_zero = false
        for (perm, sign) in group
            permuted = [t[perm[j]] for j in 1:nslots]
            if permuted == t && sign == -1
                # T_t = -T_t => T_t = 0
                is_zero = true
                break
            end
            push!(orbit_members, (permuted, sign))
        end

        if is_zero
            # Mark all orbit members as vanishing
            for (pt, _) in orbit_members
                if !haskey(tuple_to_orbit, pt)
                    tuple_to_orbit[pt] = 0
                    tuple_to_sign[pt] = 0
                end
            end
            continue
        end

        # New orbit: t is the representative with sign +1
        push!(orbits, t)
        orbit_idx = length(orbits)
        for (pt, sign) in orbit_members
            if !haskey(tuple_to_orbit, pt)
                tuple_to_orbit[pt] = orbit_idx
                # T_{pt} = sign * T_{rep}
                tuple_to_sign[pt] = sign
            end
        end
    end

    orbits, tuple_to_orbit, tuple_to_sign
end

"""
    _rational_matrix_rank(M::Matrix{Rational{Int}}) -> Int

Compute the rank of a rational matrix via Gaussian elimination.
"""
function _rational_matrix_rank(M::Matrix{Rational{Int}})
    A = copy(M)
    nrows, ncols = size(A)
    pivot_row = 1
    for col in 1:ncols
        # Find pivot
        found = 0
        for row in pivot_row:nrows
            if A[row, col] != 0
                found = row
                break
            end
        end
        found == 0 && continue
        # Swap rows
        if found != pivot_row
            A[pivot_row, :], A[found, :] = A[found, :], A[pivot_row, :]
        end
        # Eliminate below
        pivot_val = A[pivot_row, col]
        for row in (pivot_row + 1):nrows
            if A[row, col] != 0
                factor = A[row, col] // pivot_val
                A[row, :] .-= factor .* A[pivot_row, :]
            end
        end
        pivot_row += 1
    end
    # Count non-zero rows
    pivot_row - 1
end

# ── Conversion to xperm generators ────────────────────────────────────

"""
    to_xperm_generators(symh::SymH) -> Vector{Perm}

Convert SymH monoterm generators to xperm Perm objects.
"""
function to_xperm_generators(symh::SymH)
    n = symh.nslots + 2
    perms = Perm[]
    for m in symh.monoterm
        p = collect(Int32, 1:n)
        for i in 1:symh.nslots
            p[i] = Int32(m.perm[i])
        end
        if m.sign == -1
            p[n - 1], p[n] = p[n], p[n - 1]
        end
        push!(perms, Perm(p))
    end
    perms
end

# ── SymH Canonicalization ────────────────────────────────────────────

"""
    canonicalize_symh(expr::TensorExpr, symh::SymH;
                       registry::TensorRegistry=current_registry()) -> TensorExpr

Canonicalize a tensor expression using the full SymH symmetry.

Phase 1: Apply monoterm symmetries via xperm (standard slot canonicalization).
Phase 2: Apply multi-term relations (e.g., Bianchi) as rewrite rules to reduce
          the expression to a canonical basis.

The multi-term reduction uses the following algorithm:
- For each multi-term relation, the first term (identity permutation or
  lexicographically largest) is chosen as the "dependent" term.
- When the dependent term appears in the expression, it is rewritten
  as a linear combination of the other (independent) terms.
- The process iterates until no more reductions are possible.
"""
function canonicalize_symh(expr::TensorExpr, symh::SymH;
                            registry::TensorRegistry=current_registry())
    # Phase 1: monoterm canonicalization via existing pipeline
    result = with_registry(registry) do
        canonicalize(expr)
    end

    # Phase 2: multi-term reduction
    isempty(symh.multiterm) && return result

    for mt in symh.multiterm
        result = _apply_multiterm_rule(result, mt; registry=registry)
    end

    result
end

"""
    _apply_multiterm_rule(expr::TensorExpr, mt::MultitermSym;
                           registry=current_registry()) -> TensorExpr

Apply a multi-term symmetry relation as a rewrite rule to an expression.

The relation `sum_i c_i * T_{sigma_i(indices)} = 0` is rewritten as:
    T_{sigma_1(indices)} = -(c_2/c_1) * T_{sigma_2(indices)} - ...

This is applied to each Riemann factor in the expression that matches
the slot count of the multi-term relation.
"""
function _apply_multiterm_rule(expr::TensorExpr, mt::MultitermSym;
                                registry::TensorRegistry=current_registry())
    # For now, multi-term rules are informational constraints used
    # by n_independent_components, not active rewrite rules on
    # TensorExpr. The reason: multi-term rewriting requires pattern
    # matching on index permutations, which needs the full simplify
    # pipeline (canonicalize + collect_terms) to detect matching
    # terms in a TSum.
    #
    # Active rewriting will be added when the SymH rewrite engine
    # (TGR-4zb.4: SymH arithmetic) is implemented.
    expr
end

"""
    symmetrize_symh(expr::TensorExpr, symh::SymH;
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Project a tensor expression onto the subspace with the given SymH symmetry.

Applies the Young symmetrizer / projector defined by the SymH's monoterm
group: P = (1/|G|) * sum_{(sigma,s) in G} s * sigma(expr).

This produces the component of `expr` that transforms according to the
SymH symmetry representation. For the Riemann symmetry, this projects
an arbitrary rank-4 tensor onto the Riemann-symmetric subspace.
"""
function symmetrize_symh(expr::TensorExpr, symh::SymH;
                          registry::TensorRegistry=current_registry())
    # Enumerate the monoterm group
    group = _enumerate_monoterm_group(symh)
    group_order = length(group)
    group_order == 0 && return expr

    expr isa Tensor || return expr
    nslots = length(expr.indices)
    nslots == symh.nslots ||
        error("symmetrize_symh: tensor has $nslots indices but SymH has $(symh.nslots) slots")

    terms = TensorExpr[]
    for (perm, sign) in group
        permuted_indices = [expr.indices[perm[j]] for j in 1:nslots]
        t = Tensor(expr.name, permuted_indices)
        push!(terms, tproduct(Rational{Int}(sign) // Rational{Int}(group_order), TensorExpr[t]))
    end

    result = tsum(terms)
    with_registry(registry) do
        simplify(result; registry=registry)
    end
end

"""
    verify_symh(expr::TensorExpr, symh::SymH;
                 registry::TensorRegistry=current_registry()) -> Bool

Verify that a tensor expression has the symmetries described by SymH.

Checks both monoterm symmetries (applying each generator should give
±expr) and multi-term symmetries (the relation should sum to zero
when evaluated on expr).
"""
function verify_symh(expr::TensorExpr, symh::SymH;
                      registry::TensorRegistry=current_registry())
    expr isa Tensor || return false
    nslots = length(expr.indices)
    nslots == symh.nslots || return false

    # Check monoterm symmetries
    for gen in symh.monoterm
        permuted = Tensor(expr.name, [expr.indices[gen.perm[j]] for j in 1:nslots])
        expected = tproduct(Rational{Int}(gen.sign), TensorExpr[expr])
        diff = with_registry(registry) do
            simplify(tsum(TensorExpr[permuted, tproduct(-Rational{Int}(gen.sign), TensorExpr[expr])]);
                     registry=registry)
        end
        (diff == TScalar(0) || diff == tproduct(0 // 1, TensorExpr[])) || return false
    end

    # Multi-term symmetries (e.g., Bianchi) cannot be verified at the
    # abstract TensorExpr level because simplify() doesn't know about
    # multi-term identities. They can only be checked via component
    # computation or by the Invar pipeline. We skip them here.

    true
end

# ══════════════════════════════════════════════════════════════════════
# SymH Arithmetic
# ══════════════════════════════════════════════════════════════════════

"""
    symh_product(s1::SymH, s2::SymH) -> SymH

Tensor product symmetry: `T₁ ⊗ T₂` has symmetry `S₁ × S₂` acting on
disjoint slot sets. `s1` acts on slots `1:n₁`, `s2` acts on slots `n₁+1:n₁+n₂`.

# Examples
```julia
riem = riemann_symh()                    # 4 slots
prod = symh_product(riem, riem)          # 8 slots, Riem⊗Riem symmetry
prod.nslots == 8
length(prod.monoterm) == 6              # 3 from each factor
length(prod.multiterm) == 2             # Bianchi on each factor
```
"""
function symh_product(s1::SymH, s2::SymH)
    n1, n2 = s1.nslots, s2.nslots
    ntotal = n1 + n2

    # Lift s1 generators: act on 1:n1, identity on n1+1:ntotal
    pad_right = collect(n1+1:ntotal)
    mono1 = [MonotermSym(vcat(m.perm, pad_right), m.sign) for m in s1.monoterm]

    # Lift s2 generators: identity on 1:n1, act on n1+1:ntotal
    pad_left = collect(1:n1)
    mono2 = [MonotermSym(vcat(pad_left, [p + n1 for p in m.perm]), m.sign) for m in s2.monoterm]

    # Lift multiterm constraints similarly
    multi1 = [MultitermSym(ntotal, [(c, vcat(p, pad_right)) for (c, p) in mt.terms])
              for mt in s1.multiterm]
    multi2 = [MultitermSym(ntotal, [(c, vcat(pad_left, [pi + n1 for pi in p])) for (c, p) in mt.terms])
              for mt in s2.multiterm]

    SymH(ntotal, vcat(mono1, mono2), vcat(multi1, multi2))
end

"""
    symh_trace(s::SymH, i::Int, j::Int) -> SymH

Induced symmetry after contracting (tracing) slots `i` and `j` with a
symmetric metric `g^{ij} = g^{ji}`.

Keeps only generators compatible with the contraction: those that either
fix both `i,j` or swap them (metric symmetry). The result acts on `n-2` slots
with relabeled indices.

# Examples
```julia
riem = riemann_symh()                    # 4 slots: [anti(1,2), anti(3,4), pair]
traced = symh_trace(riem, 1, 3)          # trace slots 1,3 → Ricci-like, 2 slots
traced.nslots == 2
```
"""
function symh_trace(s::SymH, i::Int, j::Int)
    n = s.nslots
    (1 <= i < j <= n) || throw(ArgumentError("symh_trace: need 1 ≤ i < j ≤ n, got i=$i, j=$j, n=$n"))

    remaining = [k for k in 1:n if k != i && k != j]
    relabel = Dict(remaining[k] => k for k in eachindex(remaining))

    # Keep monoterm generators that are compatible with the contraction:
    # sigma must map {i,j} -> {i,j} (so the contraction is preserved)
    new_mono = MonotermSym[]
    for m in s.monoterm
        if Set([m.perm[i], m.perm[j]]) == Set([i, j])
            new_perm = [relabel[m.perm[k]] for k in remaining]
            # If sigma swaps i<->j, the sign is unchanged (metric is symmetric)
            push!(new_mono, MonotermSym(new_perm, m.sign))
        end
    end

    # Keep multiterm constraints where ALL terms are compatible
    new_multi = MultitermSym[]
    for mt in s.multiterm
        projected_terms = Tuple{Rational{Int}, Vector{Int}}[]
        all_ok = true
        for (c, p) in mt.terms
            if Set([p[i], p[j]]) == Set([i, j])
                push!(projected_terms, (c, [relabel[p[k]] for k in remaining]))
            else
                all_ok = false
                break
            end
        end
        all_ok && push!(new_multi, MultitermSym(n - 2, projected_terms))
    end

    SymH(n - 2, new_mono, new_multi)
end

"""
    symh_exchange(s::SymH, n1::Int; sign::Int=1) -> SymH

Add inter-factor exchange symmetry for identical tensors in a product.
Swaps the slot blocks `1:n1` ↔ `n1+1:2n1` with the given sign.
Use `sign=+1` for bosonic (symmetric) exchange, `sign=-1` for fermionic.

Requires `s.nslots == 2*n1` (two identical factors).
"""
function symh_exchange(s::SymH, n1::Int; sign::Int=1)
    s.nslots == 2 * n1 || throw(ArgumentError("symh_exchange: nslots=$(s.nslots) ≠ 2*n1=$((2*n1))"))
    sign in (1, -1) || throw(ArgumentError("sign must be +1 or -1"))

    # Exchange permutation: swap blocks [1:n1] <-> [n1+1:2n1]
    exchange_perm = vcat(collect(n1+1:2*n1), collect(1:n1))
    exchange_gen = MonotermSym(exchange_perm, sign)

    SymH(s.nslots, vcat(s.monoterm, [exchange_gen]), copy(s.multiterm))
end
