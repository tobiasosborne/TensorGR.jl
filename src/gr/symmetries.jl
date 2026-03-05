#= Symmetry specifications for tensor index slots.

Symmetries are encoded as permutation group generators for xperm.c.
For a tensor with `k` index slots, the permutations act on `n = k + 2`
points, where points `k+1` and `k+2` carry the sign:
  - k+1 ↔ k+2 means sign flip (antisymmetry)
  - k+1 → k+1, k+2 → k+2 means no sign change (symmetry)
=#

"""
    Symmetric(i, j)

Symmetry under exchange of slot positions `i` and `j` (1-indexed).
T_{...a_i...a_j...} = T_{...a_j...a_i...}
"""
struct Symmetric
    i::Int
    j::Int
end

"""
    AntiSymmetric(i, j)

Antisymmetry under exchange of slot positions `i` and `j` (1-indexed).
T_{...a_i...a_j...} = -T_{...a_j...a_i...}
"""
struct AntiSymmetric
    i::Int
    j::Int
end

"""
    PairSymmetric(i, j, k, l)

Symmetry under exchange of slot pairs (i,j) ↔ (k,l).
T_{...a_i a_j...a_k a_l...} = T_{...a_k a_l...a_i a_j...}
"""
struct PairSymmetric
    i::Int
    j::Int
    k::Int
    l::Int
end

"""
    RiemannSymmetry()

The full slot symmetry of the Riemann tensor R_{abcd}:
- AntiSymmetric(1,2), AntiSymmetric(3,4), PairSymmetric(1,2,3,4)
"""
struct RiemannSymmetry end

"""
    symmetry_generators(syms, nslots) -> Vector{Perm}

Convert high-level symmetry specs to permutation generators for xperm.c.
"""
function symmetry_generators(syms, nslots::Int)
    n = nslots + 2  # +2 for sign bits
    gens = Perm[]

    for s in syms
        push!(gens, _sym_to_perm(s, n)...)
    end
    gens
end

function _sym_to_perm(s::Symmetric, n)
    p = collect(Int32, 1:n)
    p[s.i], p[s.j] = p[s.j], p[s.i]
    # No sign flip
    [Perm(p)]
end

function _sym_to_perm(s::AntiSymmetric, n)
    p = collect(Int32, 1:n)
    p[s.i], p[s.j] = p[s.j], p[s.i]
    p[n-1], p[n] = p[n], p[n-1]  # sign flip
    [Perm(p)]
end

function _sym_to_perm(s::PairSymmetric, n)
    p = collect(Int32, 1:n)
    p[s.i], p[s.k] = p[s.k], p[s.i]
    p[s.j], p[s.l] = p[s.l], p[s.j]
    # No sign flip
    [Perm(p)]
end

function _sym_to_perm(::RiemannSymmetry, n)
    @assert n == 6 "RiemannSymmetry requires 4 slots (n=6)"
    vcat(
        _sym_to_perm(AntiSymmetric(1, 2), n),
        _sym_to_perm(AntiSymmetric(3, 4), n),
        _sym_to_perm(PairSymmetric(1, 2, 3, 4), n),
    )
end

"""
    FullySymmetric(slots::Int...)

Full symmetry under exchange of any pair among the given slot positions.
Generates adjacent transposition generators (i, i+1) for all consecutive pairs.
"""
struct FullySymmetric
    slots::Vector{Int}
end

FullySymmetric(slots::Int...) = FullySymmetric(collect(slots))

function _sym_to_perm(s::FullySymmetric, n)
    gens = Perm[]
    for i in 1:length(s.slots)-1
        p = collect(Int32, 1:n)
        p[s.slots[i]], p[s.slots[i+1]] = p[s.slots[i+1]], p[s.slots[i]]
        push!(gens, Perm(p))
    end
    gens
end

"""
    FullyAntiSymmetric(slots::Int...)

Full antisymmetry under exchange of any pair among the given slot positions.
"""
struct FullyAntiSymmetric
    slots::Vector{Int}
end

FullyAntiSymmetric(slots::Int...) = FullyAntiSymmetric(collect(slots))

function _sym_to_perm(s::FullyAntiSymmetric, n)
    gens = Perm[]
    for i in 1:length(s.slots)-1
        p = collect(Int32, 1:n)
        p[s.slots[i]], p[s.slots[i+1]] = p[s.slots[i+1]], p[s.slots[i]]
        p[n-1], p[n] = p[n], p[n-1]  # sign flip
        push!(gens, Perm(p))
    end
    gens
end
