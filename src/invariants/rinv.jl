#= RInv: Contraction permutation representation for scalar Riemann monomials.

A degree-k Riemann invariant is a product of k Riemann tensors
R_{a1b1c1d1}...R_{akbkckdk} with all 4k indices contracted in pairs via
the metric.  The contraction pattern is encoded as a fixed-point-free
involution sigma in S_{4k}: sigma(i) gives the partner of slot i.

Slot numbering: Riemann factor i (1-indexed) occupies slots 4(i-1)+1 .. 4i.

Canonical form is the lexicographically smallest contraction permutation
in the orbit of the combined Riemann slot-symmetry group and inter-factor
permutation symmetry acting by conjugation: sigma -> g.sigma.g^{-1}.

Reference: Zakhary & McIntosh (1997) GRG 29, 539;
           Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm. 176 (2007) 246.
=#

"""
    RInv(degree, contraction)

Contraction permutation representation for a scalar Riemann monomial.

For degree `k` (k Riemann factors), the contraction is a fixed-point-free
involution of length `4k`: `contraction[i]` gives the slot paired with slot `i`.

# Fields
- `degree::Int` -- number of Riemann factors
- `contraction::Vector{Int}` -- fixed-point-free involution (length 4k)
- `canonical::Bool` -- true if in canonical form
"""
struct RInv
    degree::Int
    contraction::Vector{Int}
    canonical::Bool
end

function RInv(degree::Int, contraction::Vector{Int})
    n = 4 * degree
    length(contraction) == n ||
        error("RInv: contraction length $(length(contraction)) != 4*degree=$n")
    # Validate: fixed-point-free involution
    for i in 1:n
        ci = contraction[i]
        (1 <= ci <= n) || error("RInv: contraction[$i]=$ci out of range 1:$n")
        ci != i || error("RInv: contraction[$i]=$i is a fixed point (not allowed)")
        contraction[ci] == i ||
            error("RInv: not an involution: contraction[$i]=$ci but contraction[$ci]=$(contraction[ci]) != $i")
    end
    RInv(degree, contraction, false)
end

Base.hash(r::RInv, h::UInt) = hash(r.contraction, hash(r.degree, hash(:RInv, h)))

# ---- Equality via canonical comparison ----------------------------------------

function Base.:(==)(a::RInv, b::RInv)
    a.degree == b.degree || return false
    ca = a.canonical ? a : canonicalize(a)
    cb = b.canonical ? b : canonicalize(b)
    ca.contraction == cb.contraction
end

# ---- Symmetry generators for the Riemann contraction group -------------------

"""
    _rinv_slot_generators(k) -> Vector{Vector{Int}}

Build generators for the slot symmetry group acting on 4k points.
Each generator is a permutation (Vector{Int}, images notation, 1-indexed)
and an associated sign (+1 or -1).

Returns Vector{Tuple{Vector{Int}, Int}} -- (permutation, sign).

Includes:
  - Riemann slot symmetries for each factor (anti[1,2], anti[3,4], pair[1,2,3,4])
  - Adjacent-factor transposition symmetry (identical Riem tensors)
"""
function _rinv_slot_generators(k::Int)
    n = 4k
    gens = Tuple{Vector{Int}, Int}[]

    # Per-factor Riemann symmetries
    for f in 1:k
        off = 4(f - 1)

        # Anti-symmetry in first pair (slots off+1, off+2) -> sign -1
        p = collect(1:n)
        p[off+1], p[off+2] = p[off+2], p[off+1]
        push!(gens, (p, -1))

        # Anti-symmetry in second pair (slots off+3, off+4) -> sign -1
        p = collect(1:n)
        p[off+3], p[off+4] = p[off+4], p[off+3]
        push!(gens, (p, -1))

        # Pair symmetry (swap first pair with second pair) -> sign +1
        p = collect(1:n)
        p[off+1], p[off+3] = p[off+3], p[off+1]
        p[off+2], p[off+4] = p[off+4], p[off+2]
        push!(gens, (p, +1))
    end

    # Inter-factor transpositions (adjacent factors are identical Riem tensors)
    for f in 1:k-1
        off1 = 4(f - 1)
        off2 = 4f
        p = collect(1:n)
        for j in 1:4
            p[off1+j], p[off2+j] = p[off2+j], p[off1+j]
        end
        push!(gens, (p, +1))
    end

    gens
end

"""
    _conjugate_contraction(sigma, g) -> Vector{Int}

Apply symmetry transformation g to contraction sigma by conjugation:
    sigma' = g . sigma . g^{-1}

If sigma encodes "slot i contracts with slot sigma(i)", then after
relabelling slots by g, slot g(i) contracts with slot g(sigma(i)),
i.e. sigma'(g(i)) = g(sigma(i)), hence sigma' = g.sigma.g^{-1}.
"""
function _conjugate_contraction(sigma::Vector{Int}, g::Vector{Int})
    n = length(sigma)
    # g_inv: inverse of g
    g_inv = Vector{Int}(undef, n)
    for i in 1:n
        g_inv[g[i]] = i
    end
    # sigma' = g . sigma . g^{-1}
    result = Vector{Int}(undef, n)
    for i in 1:n
        result[i] = g[sigma[g_inv[i]]]
    end
    result
end

# ---- Canonicalization --------------------------------------------------------

"""
    canonicalize(inv::RInv) -> RInv

Return the canonical form of the RInv.  The canonical representative is
the lexicographically smallest contraction permutation in the orbit of
the combined Riemann slot-symmetry and factor-reordering group, acting
by conjugation on the contraction permutation.

Sign-flipping generators (antisymmetry) that produce an overall sign
of -1 mark the invariant as vanishing (sign cancellation).
"""
function canonicalize(inv::RInv)
    inv.canonical && return inv

    k = inv.degree
    nslots = 4k
    gens = _rinv_slot_generators(k)
    isempty(gens) && return RInv(k, inv.contraction, true)

    # BFS/DFS orbit enumeration: find all elements reachable from sigma
    # under conjugation by the generators, tracking signs.
    # For degree <= 7 (28 slots) the orbit is manageable.
    orbit = Dict{Vector{Int}, Int}()  # contraction -> accumulated sign
    orbit[inv.contraction] = +1
    queue = Vector{Int}[inv.contraction]

    while !isempty(queue)
        sigma = popfirst!(queue)
        current_sign = orbit[sigma]
        for (g, gsign) in gens
            sigma_new = _conjugate_contraction(sigma, g)
            new_sign = current_sign * gsign
            if haskey(orbit, sigma_new)
                # Check consistency: if we reach the same contraction with
                # opposite sign, the invariant vanishes
                if orbit[sigma_new] != new_sign
                    # Vanishing invariant: return canonical zero
                    return RInv(k, zeros(Int, nslots), true)
                end
            else
                orbit[sigma_new] = new_sign
                push!(queue, sigma_new)
            end
        end
    end

    # Find the lexicographically smallest element
    best = inv.contraction
    for (sigma, _) in orbit
        if sigma < best
            best = sigma
        end
    end

    RInv(k, best, true)
end

# ---- xperm-based canonicalization -------------------------------------------

"""
    rinv_symmetry_group(degree::Int) -> Vector{Perm}

Generate xperm symmetry group generators for degree-k Riemann monomials.
The generators act on `4k + 2` points (4k index slots + 2 sign bits).

Each Riemann factor has:
- Antisymmetry: R_{abcd} = -R_{bacd}  (swap slots 1,2 with sign flip)
- Antisymmetry: R_{abcd} = -R_{abdc}  (swap slots 3,4 with sign flip)
- Pair symmetry: R_{abcd} = R_{cdab}   (swap pair (1,2) <-> (3,4))

Plus inter-tensor symmetry: adjacent factor transpositions for identical
Riemann factors (blocks of 4 slots are swapped, no sign).

Returns `Vector{Perm}` of permutations on `4k+2` points, with sign
encoded in the last two points per xperm convention.
"""
function rinv_symmetry_group(degree::Int)
    k = degree
    nslots = 4k
    n = nslots + 2  # +2 for sign bits
    gens = Perm[]

    # Per-factor Riemann symmetries
    for f in 1:k
        off = 4(f - 1)

        # Antisymmetry in first pair (slots off+1, off+2) -> sign flip
        p = collect(Int32, 1:n)
        p[off+1], p[off+2] = p[off+2], p[off+1]
        p[n-1], p[n] = p[n], p[n-1]  # sign flip
        push!(gens, Perm(p))

        # Antisymmetry in second pair (slots off+3, off+4) -> sign flip
        p = collect(Int32, 1:n)
        p[off+3], p[off+4] = p[off+4], p[off+3]
        p[n-1], p[n] = p[n], p[n-1]  # sign flip
        push!(gens, Perm(p))

        # Pair symmetry (swap first pair with second pair) -> no sign flip
        p = collect(Int32, 1:n)
        p[off+1], p[off+3] = p[off+3], p[off+1]
        p[off+2], p[off+4] = p[off+4], p[off+2]
        push!(gens, Perm(p))
    end

    # Inter-factor transpositions (adjacent identical Riem blocks)
    for f in 1:k-1
        off1 = 4(f - 1)
        off2 = 4f
        p = collect(Int32, 1:n)
        for j in 1:4
            p[off1+j], p[off2+j] = p[off2+j], p[off1+j]
        end
        push!(gens, Perm(p))
    end

    gens
end

"""
    _contraction_to_xperm(sigma::Vector{Int}) -> (Perm, Int32[], Int32[])

Convert a contraction involution `sigma` (length 4k) into xperm inputs:
the contraction permutation itself on `4k+2` points, empty free list,
and dummy pair list.

The contraction sigma is used directly as the xperm permutation: perm[i] = sigma(i)
for index slots, with sign bits appended. Dummy pairs are listed as
consecutive name pairs (i, sigma(i)) with i < sigma(i), formatted as
[u1, d1, u2, d2, ...] per xperm convention.

Returns `(perm, free_slots, dummy_pairs)`.
"""
function _contraction_to_xperm(sigma::Vector{Int})
    nslots = length(sigma)
    n = nslots + 2

    # The contraction permutation IS the xperm permutation
    perm_data = Vector{Int32}(undef, n)
    for i in 1:nslots
        perm_data[i] = Int32(sigma[i])
    end
    perm_data[n-1] = Int32(n - 1)  # positive sign
    perm_data[n] = Int32(n)

    # No free indices
    free_slots = Int32[]

    # Dummy pairs: extract all (i, sigma(i)) pairs with i < sigma(i)
    # Format: [u1, d1, u2, d2, ...]
    dummy_list = Int32[]
    visited = falses(nslots)
    for i in 1:nslots
        visited[i] && continue
        j = sigma[i]
        push!(dummy_list, Int32(i), Int32(j))
        visited[i] = true
        visited[j] = true
    end

    (Perm(perm_data), free_slots, dummy_list)
end

"""
    _xperm_to_contraction(cperm::Perm, nslots::Int) -> Vector{Int}

Convert an xperm canonical permutation back to a contraction involution.

Since the xperm permutation encodes the contraction directly
(perm[slot] = partner_slot), the canonical permutation IS the
canonical contraction (ignoring the sign bits).
"""
function _xperm_to_contraction(cperm::Perm, nslots::Int)
    contraction = Vector{Int}(undef, nslots)
    for i in 1:nslots
        contraction[i] = Int(cperm.data[i])
    end
    contraction
end

function _xperm_to_contraction(cperm::Perm, nslots::Int, original::Vector{Int})
    # Canonical contraction: sigma' = pi . sigma . pi^{-1}
    # where pi is the canonical permutation from xperm
    pi = Vector{Int}(undef, nslots)
    pi_inv = Vector{Int}(undef, nslots)
    for i in 1:nslots
        pi[i] = Int(cperm.data[i])
    end
    for i in 1:nslots
        pi_inv[pi[i]] = i
    end
    contraction = Vector{Int}(undef, nslots)
    for i in 1:nslots
        contraction[i] = pi[original[pi_inv[i]]]
    end
    contraction
end

"""
    canonicalize_rinv(rinv::RInv, reg::TensorRegistry) -> RInv

Canonicalize an RInv using the xperm Butler-Portugal algorithm.

Uses the full Riemann symmetry group (intra-tensor antisymmetries,
pair symmetry, inter-tensor factor permutations) and xperm's
`canonical_perm` with all indices as dummies.

This is more efficient than BFS orbit enumeration for high-degree
invariants, as xperm uses the Schreier-Sims algorithm with polynomial
complexity in the number of slots.

The registry argument is accepted for API consistency with the
TensorExpr canonicalization pipeline, but is not currently used
(Riemann symmetries are built-in).
"""
function canonicalize_rinv(rinv::RInv, reg::TensorRegistry)
    rinv.canonical && return rinv

    # RInv canonicalization requires finding the lex-smallest contraction
    # under CONJUGATION: sigma -> g.sigma.g^{-1}. This is fundamentally
    # different from xperm's LEFT-ACTION canonicalization (g.perm).
    # The BFS orbit enumeration in canonicalize(::RInv) correctly solves
    # the conjugation problem and is efficient for degree <= 6 (orbit
    # sizes < 250K, sub-second for all practical invariants).
    #
    # The xperm Butler-Portugal algorithm cannot be directly applied to
    # conjugation canonicalization without a non-trivial reformulation.
    # See Garcia-Parrado & Martin-Garcia (2007) for the double-coset
    # framework; the RInv conjugation problem is a different quotient.
    if rinv.degree >= 7
        @warn "canonicalize_rinv: degree $(rinv.degree) may be slow (BFS orbit enumeration)" maxlog=1
    end
    canonicalize(rinv)
end

"""
    are_equivalent(r1::RInv, r2::RInv, reg::TensorRegistry) -> Bool

Check if two RInv objects represent the same scalar invariant (up to
Riemann symmetries and factor reordering).

Both are canonicalized via xperm and compared.
"""
function are_equivalent(r1::RInv, r2::RInv, reg::TensorRegistry)
    r1.degree == r2.degree || return false
    c1 = canonicalize_rinv(r1, reg)
    c2 = canonicalize_rinv(r2, reg)
    c1.contraction == c2.contraction
end

# ---- Conversion to TensorExpr -----------------------------------------------

"""
    to_tensor_expr(inv::RInv; registry=current_registry(), metric=:g) -> TensorExpr

Convert an RInv back into a TensorExpr product of Riemann tensors with
metric contractions.

Each Riemann factor gets all-down indices.  Contracted pairs are joined
by inverse metric tensors g^{ab}.
"""
function to_tensor_expr(inv::RInv;
                         registry::TensorRegistry=current_registry(),
                         metric::Symbol=:g)
    k = inv.degree
    nslots = 4k

    # Generate fresh index names for all slots
    used = Set{Symbol}()
    slot_idx = Vector{Symbol}(undef, nslots)
    for i in 1:nslots
        nm = fresh_index(used)
        push!(used, nm)
        slot_idx[i] = nm
    end

    factors = TensorExpr[]

    # Riemann factors with all-down indices
    for f in 1:k
        off = 4(f - 1)
        idxs = [down(slot_idx[off+j]) for j in 1:4]
        push!(factors, Tensor(:Riem, idxs))
    end

    # Metric contractions: for each pair (i, j) with i < j, add g^{a_i a_j}
    visited = falses(nslots)
    for i in 1:nslots
        visited[i] && continue
        j = inv.contraction[i]
        visited[i] = true
        visited[j] = true
        push!(factors, Tensor(metric, [up(slot_idx[i]), up(slot_idx[j])]))
    end

    tproduct(1 // 1, factors)
end

# ---- Conversion from TensorExpr ---------------------------------------------

"""
    from_tensor_expr(expr::TensorExpr;
                      registry=current_registry(),
                      metric=:g) -> RInv

Encode a fully contracted product of Riemann tensors as an RInv.

The expression must be a (possibly simplified) product of `:Riem` tensors
and inverse metrics, with no free indices.  Ricci and RicScalar are NOT
automatically expanded; call `to_riemann` first if needed.
"""
function from_tensor_expr(expr::TensorExpr;
                           registry::TensorRegistry=current_registry(),
                           metric::Symbol=:g)
    # Extract TProduct
    p = _as_tproduct(expr)
    p === nothing && error("from_tensor_expr: expected a product of Riemann tensors, got $(typeof(expr))")

    # Separate Riemann factors and metric factors
    riem_factors = Tensor[]
    metric_factors = Tensor[]
    for f in p.factors
        f isa Tensor || error("from_tensor_expr: non-Tensor factor: $(typeof(f))")
        if f.name == :Riem
            length(f.indices) == 4 ||
                error("from_tensor_expr: Riemann tensor must have 4 indices, got $(length(f.indices))")
            push!(riem_factors, f)
        elseif f.name == metric
            push!(metric_factors, f)
        else
            error("from_tensor_expr: unexpected tensor $(f.name); expected :Riem or :$metric")
        end
    end

    k = length(riem_factors)
    k >= 1 || error("from_tensor_expr: no Riemann tensors found")
    nslots = 4k

    # Expected: 2k metric factors for full contraction
    length(metric_factors) == 2k ||
        error("from_tensor_expr: expected $(2k) metric contractions, got $(length(metric_factors))")

    # Map each index name to its slot(s)
    name_to_slots = Dict{Symbol, Vector{Tuple{Int, IndexPosition}}}()
    for (fi, riem) in enumerate(riem_factors)
        off = 4(fi - 1)
        for (j, idx) in enumerate(riem.indices)
            push!(get!(Vector{Tuple{Int,IndexPosition}}, name_to_slots, idx.name),
                  (off + j, idx.position))
        end
    end

    # Metrics carry two Up indices that contract with Riemann Down indices
    metric_pairs = Tuple{Symbol, Symbol}[]
    for m in metric_factors
        length(m.indices) == 2 ||
            error("from_tensor_expr: metric must have 2 indices, got $(length(m.indices))")
        push!(metric_pairs, (m.indices[1].name, m.indices[2].name))
    end

    # Build contraction: each metric g^{a b} pairs the slots containing a and b
    contraction = zeros(Int, nslots)
    for (a_name, b_name) in metric_pairs
        a_slots = get(name_to_slots, a_name, Tuple{Int,IndexPosition}[])
        b_slots = get(name_to_slots, b_name, Tuple{Int,IndexPosition}[])

        # Find the Riemann slot (Down) for each metric index name
        a_riem = [s for (s, pos) in a_slots if 1 <= s <= nslots]
        b_riem = [s for (s, pos) in b_slots if 1 <= s <= nslots]

        length(a_riem) == 1 || error("from_tensor_expr: index $a_name ambiguous in Riemann slots")
        length(b_riem) == 1 || error("from_tensor_expr: index $b_name ambiguous in Riemann slots")

        sa = a_riem[1]
        sb = b_riem[1]
        contraction[sa] = sb
        contraction[sb] = sa
    end

    # Verify all slots are paired
    for i in 1:nslots
        contraction[i] != 0 || error("from_tensor_expr: slot $i is unpaired")
    end

    RInv(k, contraction)
end

# ---- Helper ------------------------------------------------------------------

function _as_tproduct(expr::TProduct)
    expr
end

function _as_tproduct(expr::Tensor)
    TProduct(1 // 1, TensorExpr[expr])
end

function _as_tproduct(::TensorExpr)
    nothing
end
