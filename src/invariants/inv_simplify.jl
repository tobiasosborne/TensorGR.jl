#= InvSimplify: Database-driven simplification of scalar Riemann invariants.
#
# Fast path for the Invar pipeline: instead of running the full 6-level
# algorithm, look up precomputed reduction relations in the database.
# Falls back to `riemann_simplify` for expressions not in the database.
#
# Algorithm:
#   1. Convert expr to RInv (contraction permutation form)
#   2. Canonicalize the RInv
#   3. Look up in the database:
#      a. Independent invariant -> return canonical form
#      b. Dependent invariant -> return RHS linear combination
#      c. Not in database -> fall back to riemann_simplify
#   4. Convert result back to TensorExpr
#
# Reference: Garcia-Parrado & Martin-Garcia, Comp. Phys. Comm. 176 (2007) 246, Sec 5.
=#

"""
    inv_simplify(expr::TensorExpr;
                  registry::TensorRegistry=current_registry(),
                  dim::Union{Int,Nothing}=nothing,
                  metric::Symbol=:g) -> TensorExpr

Simplify a scalar Riemann invariant using precomputed database lookup.

Fast path: converts to RInv canonical form and looks up in the Invar
database. If a reduction relation exists, returns the simplified form
as a linear combination of independent invariants. Falls back to
`riemann_simplify` for expressions not in the database.

# Arguments
- `expr`: a scalar curvature expression (product of Riemann tensors)
- `registry`: tensor registry
- `dim`: manifold dimension (for DDI-level lookups)
- `metric`: metric tensor name

# Returns
Simplified TensorExpr in terms of independent invariants.

Ground truth: Garcia-Parrado & Martin-Garcia (2007) Sec 5.
"""
function inv_simplify(expr::TensorExpr;
                       registry::TensorRegistry=current_registry(),
                       dim::Union{Int,Nothing}=nothing,
                       metric::Symbol=:g)
    # TScalar pass-through
    expr isa TScalar && return expr

    # Handle TSum: simplify each term individually and re-sum
    if expr isa TSum
        simplified_terms = TensorExpr[
            inv_simplify(t; registry=registry, dim=dim, metric=metric)
            for t in expr.terms
        ]
        result = tsum(simplified_terms)
        # Collect terms after simplifying to combine like invariants
        return with_registry(registry) do
            collect_terms(result)
        end
    end

    # Try to convert to RInv
    rinv = _try_to_rinv(expr; registry=registry, metric=metric)

    if rinv !== nothing
        # Canonicalize
        canon = canonicalize(rinv)

        # Check if it's zero (vanishing invariant)
        if all(==(0), canon.contraction)
            return TScalar(0 // 1)
        end

        degree = canon.degree
        case_key = _algebraic_case_key(degree)

        # Try database lookup at Level 2 (Bianchi)
        step = 2
        key = (degree, case_key, step, dim)
        cr = nothing
        if haskey(_INVAR_DB_REGISTRY, key)
            cr = _INVAR_DB_REGISTRY[key]
        end
        # Also try dimension-independent if dim-specific not found
        if cr === nothing && dim !== nothing
            key_nodim = (degree, case_key, step, nothing)
            if haskey(_INVAR_DB_REGISTRY, key_nodim)
                cr = _INVAR_DB_REGISTRY[key_nodim]
            end
        end

        if cr !== nothing
            # Extract the scalar coefficient from the original expression
            coeff = _extract_rinv_coefficient(expr)

            # Check if this contraction is dependent (has a relation)
            for rel in cr.relations
                if rel.lhs == canon.contraction
                    # Found a relation! Build the RHS as TensorExpr
                    rhs = _build_rhs_expr(rel.rhs; registry=registry, metric=metric, degree=degree)
                    # Apply the original coefficient
                    if coeff != 1 // 1
                        return tproduct(coeff, TensorExpr[rhs])
                    end
                    return rhs
                end
            end
            # Not in relations -> it's independent, return canonical form
            canon_expr = to_tensor_expr(canon; registry=registry, metric=metric)
            if coeff != 1 // 1
                return tproduct(coeff, TensorExpr[canon_expr])
            end
            return canon_expr
        end
    end

    # Fallback: use riemann_simplify
    riemann_simplify(expr; registry=registry, dim=dim)
end

"""
    _try_to_rinv(expr; registry, metric) -> Union{RInv, Nothing}

Attempt to convert a TensorExpr to RInv form. Returns `nothing` on failure
(e.g., if the expression is not a product of Riemann tensors with metrics).
"""
function _try_to_rinv(expr::TensorExpr;
                       registry::TensorRegistry=current_registry(),
                       metric::Symbol=:g)
    # Strip scalar coefficient from TProduct before converting
    inner = _strip_coefficient(expr)
    try
        from_tensor_expr(inner; registry=registry, metric=metric)
    catch
        nothing
    end
end

"""
    _strip_coefficient(expr::TensorExpr) -> TensorExpr

Strip the scalar coefficient from a TProduct, returning a TProduct with
coefficient 1//1 but the same factors. For non-TProduct expressions,
returns the expression unchanged.
"""
function _strip_coefficient(expr::TProduct)
    if expr.scalar == 1 // 1
        return expr
    end
    TProduct(1 // 1, expr.factors)
end

function _strip_coefficient(expr::TensorExpr)
    expr
end

"""
    _extract_rinv_coefficient(expr::TensorExpr) -> Rational{Int}

Extract the scalar coefficient from a TProduct. Returns 1//1 for
non-TProduct expressions.
"""
function _extract_rinv_coefficient(expr::TProduct)
    expr.scalar
end

function _extract_rinv_coefficient(::TensorExpr)
    1 // 1
end

"""
    _build_rhs_expr(rhs; registry, metric, degree) -> TensorExpr

Convert the RHS of an InvarRelation (a linear combination of canonical
contraction permutations) to a TensorExpr.

The RHS format is `Vector{Tuple{Rational{Int}, Vector{Int}}}` where each
tuple is `(coefficient, canonical_contraction)`.
"""
function _build_rhs_expr(rhs::Vector{Tuple{Rational{Int}, Vector{Int}}};
                          registry::TensorRegistry=current_registry(),
                          metric::Symbol=:g,
                          degree::Int=0)
    if length(rhs) == 1
        coeff, contraction = rhs[1]
        rinv = RInv(degree, contraction, true)
        texpr = to_tensor_expr(rinv; registry=registry, metric=metric)
        if coeff == 1 // 1
            return texpr
        end
        return tproduct(coeff, TensorExpr[texpr])
    end

    # Multiple terms: build a TSum
    terms = TensorExpr[]
    for (coeff, contraction) in rhs
        rinv = RInv(degree, contraction, true)
        texpr = to_tensor_expr(rinv; registry=registry, metric=metric)
        if coeff == 1 // 1
            push!(terms, texpr)
        else
            push!(terms, tproduct(coeff, TensorExpr[texpr]))
        end
    end
    tsum(terms)
end
