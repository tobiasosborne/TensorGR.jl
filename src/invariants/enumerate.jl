#= enumerate.jl: Independent basis enumeration for scalar Riemann invariants.
#
# Provides `enumerate_independent_rinvs` which returns the canonical forms,
# independent basis, and reduction relations at a given degree and
# simplification level. Uses the precomputed Invar database (db/*.jl) for
# syzygy reduction, with optional live algorithmic enumeration for
# cross-validation.
#
# Reference: Fulling, King, Wybourne & Cummins (1992), CQG 9:1151;
#            Garcia-Parrado & Martin-Garcia (2007), Comp. Phys. Comm. 176:246;
#            Zakhary & McIntosh (1997), GRG 29:539.
=#

"""
    enumerate_independent_rinvs(degree::Int; level::Int=2, dim=nothing)
        -> (canonical::Vector{RInv}, independent::Vector{RInv},
            relations::Vector{InvarRelation})

Enumerate canonical and independent scalar Riemann invariants at a given
degree and simplification level.

For degree `k`, a scalar Riemann invariant is a product of `k` Riemann
tensors with all `4k` indices contracted pairwise via the metric. The
canonical forms are the distinct contraction patterns after applying
Riemann permutation symmetries (Level 1). The independent forms are the
subset remaining after applying Bianchi identity relations (Level 2).

# Arguments
- `degree::Int` -- number of Riemann factors (≥ 1)
- `level::Int=2` -- simplification level:
    - `1`: permutation symmetries only (canonical forms)
    - `2`: + first Bianchi identity (independent basis)
- `dim::Union{Int,Nothing}=nothing` -- manifold dimension (for future DDI support)

# Returns
Named tuple with:
- `canonical::Vector{RInv}` -- all non-vanishing canonical forms
- `independent::Vector{RInv}` -- independent (non-reducible) subset
- `relations::Vector{InvarRelation}` -- reduction rules for dependent invariants

# Ground Truth (algebraic, dimension-independent)
| Degree | Canonical (L1) | Independent (L2) | Bianchi relations |
|--------|----------------|------------------|-------------------|
| 2      | 4              | 3                | 1                 |
| 3      | 13             | 8                | 5                 |
| 4      | 57             | 26               | 31                |

# Examples
```julia
reg = TensorRegistry()
with_registry(reg) do
    @manifold M4 dim=4 metric=g
    define_curvature_tensors!(reg, :M4, :g)
    result = enumerate_independent_rinvs(2)
    length(result.canonical)    # 4
    length(result.independent)  # 3
    length(result.relations)    # 1
end
```

Reference: Fulling et al. (1992), CQG 9:1151, Tables 1-2.
"""
function enumerate_independent_rinvs(degree::Int;
                                      level::Int=2,
                                      dim::Union{Int,Nothing}=nothing)
    degree >= 2 || throw(ArgumentError(
        "degree must be ≥ 2 (degree 1 has 0 independent invariants), got $degree"))
    1 <= level <= 2 || throw(ArgumentError(
        "level must be 1 or 2 (higher levels not yet supported), got $level"))

    case_key = join(fill("0", degree), "_")

    # Get all canonical forms from named accessors (source of truth)
    all_canonical = _get_canonical_rinvs(degree)

    if level == 1
        return (canonical=all_canonical, independent=copy(all_canonical),
                relations=InvarRelation[])
    end

    # Level 2: load Bianchi reduction relations
    cr2 = get_invar_relations(degree, case_key, 2; dim=dim)
    dependent_set = Set{Vector{Int}}(rel.lhs for rel in cr2.relations)
    independent = RInv[r for r in all_canonical if r.contraction ∉ dependent_set]

    return (canonical=all_canonical, independent=independent,
            relations=cr2.relations)
end

"""
    _get_canonical_rinvs(degree::Int) -> Vector{RInv}

Return all canonical RInv forms at a given degree using the named
accessor functions from the precomputed database.
"""
function _get_canonical_rinvs(degree::Int)
    if degree == 2
        return degree2_canonical_rinvs()
    elseif degree == 3
        return degree3_canonical_rinvs()
    elseif degree == 4
        return degree4_canonical_rinvs()
    else
        error("Canonical RInv accessor not available for degree $degree. " *
              "Use enumerate_live_canonical_rinvs() to compute them.")
    end
end

"""
    enumerate_live_canonical_rinvs(degree::Int; kwargs...) -> Vector{RInv}

Algorithmically enumerate all canonical RInv forms at a given degree by:
1. Generating all `(4k-1)!!` perfect matchings of `4k` Riemann slots
2. Constructing each as an RInv contraction involution
3. Canonicalizing and deduplicating

This is the live computation path for cross-validation against the
precomputed database. Feasible for degrees 2-3; degree 4 has ~2M
pairings and may be slow.

# Arguments
- `degree::Int` -- number of Riemann factors (recommended ≤ 3)

# Returns
- `Vector{RInv}` -- all non-vanishing canonical forms (sorted lexicographically)
"""
function enumerate_live_canonical_rinvs(degree::Int;
                                        registry::TensorRegistry=current_registry(),
                                        metric::Symbol=:g)
    degree >= 1 || throw(ArgumentError("degree must be ≥ 1"))
    n = 4 * degree

    # Generate all (n-1)!! perfect matchings of n elements
    pairings = _all_perfect_matchings(n)

    # Convert each pairing to an RInv, canonicalize, deduplicate
    seen = Set{Vector{Int}}()
    result = RInv[]
    for pairing in pairings
        # Skip pairings that contract antisymmetric Riemann slots
        # (vanish by R_{[ab]cd} antisymmetry)
        _pairing_vanishes(pairing, degree) && continue
        contraction = _pairing_to_involution(pairing, n)
        rinv = try
            RInv(degree, contraction)
        catch e
            e isa ErrorException || rethrow()
            continue  # invalid involution (e.g. fixed point)
        end
        canon = canonicalize(rinv)
        # Skip vanishing forms (canonical form is all zeros or has fixed points)
        all(==(0), canon.contraction) && continue
        any(i -> canon.contraction[i] == i, 1:n) && continue
        if canon.contraction ∉ seen
            push!(seen, canon.contraction)
            push!(result, canon)
        end
    end
    sort!(result; by=r -> r.contraction)
    return result
end

"""
    _all_perfect_matchings(n::Int) -> Vector{Vector{Tuple{Int,Int}}}

Generate all (n-1)!! perfect matchings of the set {1,...,n}.
Each matching is a vector of n/2 pairs (i,j) with i < j.
"""
function _all_perfect_matchings(n::Int)
    n % 2 == 0 || error("n must be even")
    result = Vector{Vector{Tuple{Int,Int}}}()
    _pm_recurse!(result, Tuple{Int,Int}[], collect(1:n))
    return result
end

function _pm_recurse!(result, current, remaining)
    if isempty(remaining)
        push!(result, copy(current))
        return
    end
    first = remaining[1]
    rest = remaining[2:end]
    for (i, partner) in enumerate(rest)
        push!(current, (first, partner))
        new_rest = [rest[j] for j in eachindex(rest) if j != i]
        _pm_recurse!(result, current, new_rest)
        pop!(current)
    end
end

"""
    _pairing_vanishes(pairing, degree) -> Bool

Check if a perfect matching vanishes due to Riemann antisymmetry.
A pairing vanishes if any pair (i,j) contracts two slots within the
same antisymmetric pair of a Riemann factor: slots (4k-3,4k-2) or
(4k-1,4k) for factor k.
"""
function _pairing_vanishes(pairing::Vector{Tuple{Int,Int}}, degree::Int)
    for (i, j) in pairing
        a, b = minmax(i, j)
        # Check if a,b are consecutive and form an antisymmetric pair
        # within the same Riemann factor: (1,2), (3,4), (5,6), (7,8), ...
        if b == a + 1 && isodd(a)
            # Both in same factor? Factor k has slots 4(k-1)+1 .. 4k
            factor_a = div(a - 1, 4) + 1
            factor_b = div(b - 1, 4) + 1
            if factor_a == factor_b
                return true
            end
        end
    end
    return false
end

"""
    _pairing_to_involution(pairing, n) -> Vector{Int}

Convert a perfect matching (list of pairs) to a fixed-point-free
involution vector of length n.
"""
function _pairing_to_involution(pairing::Vector{Tuple{Int,Int}}, n::Int)
    inv = zeros(Int, n)
    for (i, j) in pairing
        inv[i] = j
        inv[j] = i
    end
    return inv
end
