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
