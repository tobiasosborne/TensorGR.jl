#= Invar six-level simplification algorithm -- Level 1.
#
# Level 1: Permutation symmetries for Riemann monomials.
# Ensures canonical ordering of indices in products of curvature tensors
# using the existing Butler-Portugal (xperm) canonicalization.
#
# The six levels are (Garcia-Parrado & Martin-Garcia 2007, Sec 4):
#   1. Permutation symmetries (Riemann pair-swap, antisymmetry)
#   2. Cyclic symmetry (first Bianchi identity)
#   3. Dimensionally-dependent identities (DDIs)
#   4. Product symmetries (commutativity of tensor products)
#   5. Renaming symmetries (dummy index relabeling)
#   6. Schouten/Lovelock identities
#
# Level 1 is handled by the existing canonicalize() via xperm.c,
# which already knows about Riemann symmetries. This module provides
# a thin wrapper and targeted verification tests.
#
# Ground truth: Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm.
#              176 (2007) 246, Section 4, Level 1.
=#

"""
    simplify_level1(expr::TensorExpr; registry::TensorRegistry=current_registry())
        -> TensorExpr

Apply Level 1 (permutation symmetries) of the Invar simplification algorithm.

This delegates to the existing `canonicalize()` function, which uses
Butler-Portugal (xperm.c) to find the canonical form under:
- Riemann pair symmetry: R_{abcd} = R_{cdab}
- Riemann antisymmetry: R_{abcd} = -R_{bacd} = -R_{abdc}
- Ricci symmetry: R_{ab} = R_{ba}
- Metric symmetry: g_{ab} = g_{ba}
- Products: reorder factors by canonical index pattern

For Riemann monomials (products of n Riemann tensors with contracted
indices), this is the first and most basic simplification step.

Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4, Level 1.
"""
function simplify_level1(expr::TensorExpr;
                          registry::TensorRegistry=current_registry())
    with_registry(registry) do
        canonicalize(expr)
    end
end

"""
    is_riemann_monomial(expr::TensorExpr) -> Bool

Check if an expression is a scalar Riemann monomial: a product of
curvature tensors (Riemann, Ricci, RicScalar, Weyl) with all indices
contracted (no free indices).

This is the class of expressions that the Invar algorithm operates on.
"""
function is_riemann_monomial(expr::TensorExpr)
    # Must have no free indices (scalar)
    !isempty(free_indices(expr)) && return false

    # Must be a product or single curvature tensor
    _is_curvature_product(expr)
end

function _is_curvature_product(expr::Tensor)
    expr.name in (:Riem, :Ric, :RicScalar, :Weyl, :Ein, :Schouten) ||
        expr.name in (:g, :delta)  # metrics/deltas allowed in contractions
end

_is_curvature_product(::TScalar) = true  # scalar coefficients ok

function _is_curvature_product(p::TProduct)
    all(_is_curvature_product, p.factors)
end

function _is_curvature_product(s::TSum)
    all(_is_curvature_product, s.terms)
end

_is_curvature_product(::TDeriv) = false  # derivatives not in Invar scope
_is_curvature_product(::TensorExpr) = false

"""
    count_riemann_degree(expr::TensorExpr) -> Int

Count the number of Riemann/Weyl tensor factors in a monomial.
Ricci counts as 1 (trace of Riemann). RicScalar counts as 1.

This is the "degree" of the invariant in the Invar classification.
"""
function count_riemann_degree(expr::Tensor)
    expr.name in (:Riem, :Ric, :RicScalar, :Weyl, :Ein, :Schouten) ? 1 : 0
end

count_riemann_degree(::TScalar) = 0

function count_riemann_degree(p::TProduct)
    sum(count_riemann_degree(f) for f in p.factors; init=0)
end

function count_riemann_degree(s::TSum)
    isempty(s.terms) ? 0 : maximum(count_riemann_degree(t) for t in s.terms)
end

count_riemann_degree(d::TDeriv) = count_riemann_degree(d.arg)

# ────────────────────────────────────────────────────────────────────
# Level 2: Cyclic symmetry (first Bianchi identity)
# ────────────────────────────────────────────────────────────────────

#= The first Bianchi identity: R_{a[bcd]} = 0, equivalently:
#    R_{abcd} + R_{acdb} + R_{adbc} = 0
#
# For a product of n Riemann tensors, applying this identity to any
# factor generates a linear relation among monomials. Collecting all
# such relations and reducing yields the independent basis.
#
# At degree 2 in d=4: 6 naive contractions reduce to 3 independent
# invariants (R², R_{ab}R^{ab}, R_{abcd}R^{abcd}).
#
# Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4.1.
=#

"""
    apply_bianchi_cyclic(expr::TensorExpr, factor_idx::Int;
                          registry::TensorRegistry=current_registry())
        -> TensorExpr

Apply the first Bianchi identity R_{abcd} + R_{acdb} + R_{adbc} = 0
to the `factor_idx`-th Riemann factor in a product.

Returns the expression with the cycled terms subtracted:
    expr |_{R_{abcd}} -> -expr|_{R_{acdb}} - expr|_{R_{adbc}}

The result is then Level-1 canonicalized.
"""
function apply_bianchi_cyclic(expr::TensorExpr, factor_idx::Int;
                               registry::TensorRegistry=current_registry())
    expr isa TProduct || return expr

    factors = collect(expr.factors)
    (1 <= factor_idx <= length(factors)) ||
        error("factor_idx $factor_idx out of range [1, $(length(factors))]")

    riem = factors[factor_idx]
    riem isa Tensor && riem.name == :Riem || return expr

    idxs = riem.indices
    length(idxs) == 4 || return expr

    a, b, c, d = idxs

    # Original: R_{abcd}
    # Bianchi: R_{abcd} = -R_{acdb} - R_{adbc}
    riem_cycled1 = Tensor(:Riem, [a, c, d, b])
    riem_cycled2 = Tensor(:Riem, [a, d, b, c])

    other_factors = TensorExpr[factors[i] for i in 1:length(factors) if i != factor_idx]

    term1 = tproduct(-expr.scalar, TensorExpr[vcat(other_factors, [riem_cycled1])...])
    term2 = tproduct(-expr.scalar, TensorExpr[vcat(other_factors, [riem_cycled2])...])

    result = tsum(TensorExpr[term1, term2])

    # Level-1 canonicalize
    with_registry(registry) do
        canonicalize(result)
    end
end

"""
    simplify_level2(expr::TensorExpr;
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Apply Level 2 (cyclic symmetry / first Bianchi identity) of the Invar
simplification algorithm.

For each Riemann factor in each term, applies the Bianchi identity
and collects terms. This may reduce the number of independent invariants.

At degree 2 in d=4: reduces from 6 naive to 3 independent invariants.

Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 4.1, Level 2.
"""
function simplify_level2(expr::TensorExpr;
                          registry::TensorRegistry=current_registry())
    # First apply Level 1
    expr1 = simplify_level1(expr; registry=registry)

    # Then simplify via the full pipeline which includes Bianchi rules
    with_registry(registry) do
        simplify(expr1; registry=registry)
    end
end

"""
    bianchi_relation(a::TIndex, b::TIndex, c::TIndex, d::TIndex) -> TensorExpr

Construct the first Bianchi identity as an expression:

    R_{abcd} + R_{acdb} + R_{adbc} = 0

Returns the LHS (which should simplify to zero).
"""
function bianchi_relation(a::TIndex, b::TIndex, c::TIndex, d::TIndex)
    R1 = Tensor(:Riem, [a, b, c, d])
    R2 = Tensor(:Riem, [a, c, d, b])
    R3 = Tensor(:Riem, [a, d, b, c])
    tsum(TensorExpr[R1, R2, R3])
end
