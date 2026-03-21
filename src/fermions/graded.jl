#= Grassmann parity tracking for fermion fields.
#
# Fermion fields are Grassmann-valued (anticommuting). In a product of
# fields, swapping two fermionic factors produces a sign flip:
#
#     psi * chi = -(chi * psi)
#
# This module provides:
#   - register_grassmann_field!  -- register a Grassmann-odd (fermionic) tensor
#   - is_grassmann               -- check if a tensor is Grassmann-odd
#   - grassmann_parity           -- compute total Grassmann parity of an expression
#   - grassmann_sign             -- sign from permuting Grassmann-odd factors
#
# The actual sign-tracking during canonicalization is deferred to a later issue.
# This module only provides the metadata and parity queries.
#
# Ground truth:
#   Peskin & Schroeder (1995), Sec 9.5 (path integrals for fermions).
#   Dreiner, Haber & Martin (2010), arXiv:0812.1594, Sec 2.
=#

"""
    register_grassmann_field!(reg, name; manifold, rank, symmetries, options)

Register a Grassmann-odd (fermionic) tensor field in the registry.

Sets `:is_grassmann => true` in the tensor's `options` Dict. The `rank` is
specified as a tuple `(contravariant, covariant)` following the `TensorProperties`
convention.

# Example
```julia
reg = TensorRegistry()
register_manifold!(reg, ManifoldProperties(:M4, 4, :g, nothing, [:a,:b,:c,:d]))
register_grassmann_field!(reg, :psi; manifold=:M4, rank=(0,1))
is_grassmann(reg, :psi)  # => true
```
"""
function register_grassmann_field!(reg::TensorRegistry, name::Symbol;
                                    manifold::Symbol=:M4,
                                    rank::Tuple{Int,Int}=(0,1),
                                    symmetries::Vector{SymmetrySpec}=SymmetrySpec[],
                                    options::Dict{Symbol,Any}=Dict{Symbol,Any}())
    opts = copy(options)
    opts[:is_grassmann] = true
    register_tensor!(reg, TensorProperties(
        name=name, manifold=manifold, rank=rank,
        symmetries=symmetries, options=opts))
end

"""
    is_grassmann(reg::TensorRegistry, name::Symbol) -> Bool

Check if a tensor is Grassmann-odd (fermionic) by looking up `:is_grassmann`
in its `options` Dict.

Returns `false` if the tensor is not registered or has no Grassmann flag.
"""
function is_grassmann(reg::TensorRegistry, name::Symbol)
    has_tensor(reg, name) || return false
    tp = get_tensor(reg, name)
    get(tp.options, :is_grassmann, false)::Bool
end

"""
    grassmann_parity(expr::TensorExpr; registry=current_registry()) -> Int

Compute the total Grassmann parity of an expression.
Returns 0 (even/bosonic) or 1 (odd/fermionic).

Rules:
- `Tensor` with `is_grassmann=true` in registry: parity 1
- `TProduct`: sum of factor parities mod 2
- `TSum`: parity of first term (all terms must have same parity)
- `TScalar`: parity 0
- `TDeriv`: parity of argument (derivative does not change Grassmann parity)
- `GammaMatrix`: parity 0 (matrix, not a field)
"""
function grassmann_parity(expr::TensorExpr;
                           registry::TensorRegistry=current_registry())
    _grassmann_parity(expr, registry)
end

function _grassmann_parity(t::Tensor, reg::TensorRegistry)
    is_grassmann(reg, t.name) ? 1 : 0
end

function _grassmann_parity(p::TProduct, reg::TensorRegistry)
    s = 0
    for f in p.factors
        s += _grassmann_parity(f, reg)
    end
    s % 2
end

function _grassmann_parity(s::TSum, reg::TensorRegistry)
    isempty(s.terms) && return 0
    _grassmann_parity(s.terms[1], reg)
end

function _grassmann_parity(::TScalar, ::TensorRegistry)
    0
end

function _grassmann_parity(d::TDeriv, reg::TensorRegistry)
    _grassmann_parity(d.arg, reg)
end

# GammaMatrix is a matrix, not a field — parity 0
function _grassmann_parity(::GammaMatrix, ::TensorRegistry)
    0
end

# Fallback for any other TensorExpr subtypes
function _grassmann_parity(::TensorExpr, ::TensorRegistry)
    0
end

"""
    grassmann_sign(perm::Vector{Int}, factors::Vector{TensorExpr};
                    registry=current_registry()) -> Int

Compute the sign arising from permuting factors according to `perm`,
accounting for Grassmann parity. Returns +1 or -1.

`perm` is a permutation such that `factors[perm[i]]` gives the reordered
sequence. Only transpositions of two Grassmann-odd factors contribute a
sign flip.

The sign is `(-1)^k` where `k` is the number of transpositions of
Grassmann-odd factors in a bubble-sort decomposition of `perm`.
"""
function grassmann_sign(perm::Vector{Int}, factors::Vector{TensorExpr};
                         registry::TensorRegistry=current_registry())
    n = length(perm)
    n == length(factors) || error("perm length must match factors length")

    # Compute parity of each factor
    parities = [_grassmann_parity(f, registry) for f in factors]

    # Count transpositions of odd-parity factors via bubble sort
    # Work on a mutable copy of the permutation
    arr = copy(perm)
    par = copy(parities)
    sign = 1
    for i in 1:n
        for j in 1:(n - i)
            if arr[j] > arr[j + 1]
                # Swap in permutation
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                # If both are Grassmann-odd, flip sign
                if par[j] == 1 && par[j + 1] == 1
                    sign = -sign
                end
                # Swap parities to track which factor is where
                par[j], par[j + 1] = par[j + 1], par[j]
            end
        end
    end
    sign
end
