#= Generalized Kronecker delta and dimensionally-dependent identities (DDIs).

The generalized Kronecker delta is defined as:

    delta^{a1...ap}_{b1...bp} = det | delta^{a1}_{b1} ... delta^{a1}_{bp} |
                                     | ...                                 |
                                     | delta^{ap}_{b1} ... delta^{ap}_{bp} |

Equivalently, it is the antisymmetrized product of p ordinary Kronecker deltas:

    delta^{a1...ap}_{b1...bp} = p! delta^{[a1}_{b1} ... delta^{ap]}_{bp}

Key DDI property: when p > dim(manifold), the generalized delta VANISHES
identically because one cannot antisymmetrize more indices than there are
dimensions.  This is the master dimensionally-dependent identity.

References:
  - Lovelock (1971), J. Math. Phys. 12, 498
  - Fulling et al. (1992), Class. Quantum Grav. 9, 1151, Sec 6
=#

"""
    generalized_delta(up_indices, down_indices; registry=current_registry()) -> TensorExpr

Construct the generalized Kronecker delta delta^{a1...ap}_{b1...bp} as a sum
of signed products of ordinary Kronecker deltas (the Leibniz/determinant formula).

Returns `TScalar(0//1)` when `p > dim` -- this is the master DDI.

# Arguments
- `up_indices::Vector{Symbol}`: contravariant index names (length p)
- `down_indices::Vector{Symbol}`: covariant index names (length p)
- `registry`: the TensorRegistry to use (for manifold dimension and delta name lookup)

# Examples
```julia
# p=2: delta^{ab}_{cd} = delta^a_c delta^b_d - delta^a_d delta^b_c
generalized_delta([:a, :b], [:c, :d])

# p=5 in d=4: vanishes by DDI
generalized_delta([:a, :b, :c, :d, :e], [:f, :g, :h, :i, :j])  # => TScalar(0//1)
```
"""
function generalized_delta(up_indices::Vector{Symbol}, down_indices::Vector{Symbol};
                            registry::TensorRegistry=current_registry())
    p = length(up_indices)
    length(down_indices) == p || error("generalized_delta: up and down index lists must have equal length (got $p and $(length(down_indices)))")
    p == 0 && return TScalar(1 // 1)

    # Determine manifold dimension from the registry (use the first registered manifold)
    dim = _lookup_dim(registry)

    # Master DDI: vanishes when p > dim
    p > dim && return TScalar(0 // 1)

    # Find the delta tensor name
    delta_name = _lookup_delta_name(registry)

    # Expand as sum over permutations of down indices (Leibniz formula):
    # delta^{a1...ap}_{b1...bp} = sum_{sigma in S_p} sign(sigma) prod_i delta^{ai}_{b_{sigma(i)}}
    _expand_generalized_delta(up_indices, down_indices, delta_name)
end

"""
    generalized_delta(p, dim; registry=current_registry()) -> TensorExpr

Construct a rank-2p generalized Kronecker delta with fresh index names.

Returns `TScalar(0//1)` when `p > dim` (DDI).

# Arguments
- `p::Int`: number of upper (and lower) index slots
- `dim::Int`: manifold dimension (used for DDI check; overrides registry lookup)
- `registry`: TensorRegistry for delta name lookup

# Examples
```julia
generalized_delta(2, 4)  # rank-4 delta with fresh indices
generalized_delta(5, 4)  # => TScalar(0//1) by DDI
```
"""
function generalized_delta(p::Int, dim::Int; registry::TensorRegistry=current_registry())
    p < 0 && error("generalized_delta: p must be non-negative (got $p)")
    p == 0 && return TScalar(1 // 1)

    # Master DDI
    p > dim && return TScalar(0 // 1)

    # Generate fresh index names
    used = Set{Symbol}()
    up_names = Symbol[]
    down_names = Symbol[]
    for _ in 1:p
        u = fresh_index(used)
        push!(used, u)
        push!(up_names, u)
    end
    for _ in 1:p
        d = fresh_index(used)
        push!(used, d)
        push!(down_names, d)
    end

    delta_name = _lookup_delta_name(registry)
    _expand_generalized_delta(up_names, down_names, delta_name)
end

"""
    is_zero_by_dimension(p::Int, dim::Int) -> Bool

Check whether a generalized Kronecker delta of order p vanishes in dimension dim.
This is true when p > dim (the DDI condition).
"""
is_zero_by_dimension(p::Int, dim::Int) = p > dim

# ── Internal helpers ──────────────────────────────────────────────────

"""
    _expand_generalized_delta(up_names, down_names, delta_name) -> TensorExpr

Expand the generalized delta as a sum of signed products of ordinary deltas
using the Leibniz/determinant formula:

    sum_{sigma in S_p} sign(sigma) prod_i delta^{ai}_{b_{sigma(i)}}
"""
function _expand_generalized_delta(up_names::Vector{Symbol}, down_names::Vector{Symbol},
                                    delta_name::Symbol)
    p = length(up_names)

    # p=1: just an ordinary delta
    if p == 1
        return Tensor(delta_name, [up(up_names[1]), down(down_names[1])])
    end

    perms = _permutations_with_sign(p)
    terms = TensorExpr[]

    for (perm, sgn) in perms
        factors = TensorExpr[]
        for i in 1:p
            push!(factors, Tensor(delta_name, [up(up_names[i]), down(down_names[perm[i]])]))
        end
        push!(terms, tproduct(Rational{Int}(sgn), factors))
    end

    tsum(terms)
end

"""Look up manifold dimension from registry (first registered manifold)."""
function _lookup_dim(reg::TensorRegistry)
    isempty(reg.manifolds) && error("generalized_delta: no manifolds registered; cannot determine dimension")
    first(values(reg.manifolds)).dim
end

"""Look up the delta tensor name from registry."""
function _lookup_delta_name(reg::TensorRegistry)
    isempty(reg.delta_cache) ? :δ : first(values(reg.delta_cache))
end
