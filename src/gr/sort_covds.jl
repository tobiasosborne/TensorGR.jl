#= Sort covariant derivatives: commutation rules.

When commuting ∇_a ∇_b, Riemann curvature terms appear:
  [∇_a, ∇_b] V^c = R^c_{dab} V^d
  [∇_a, ∇_b] ω_c = -R^d_{cab} ω_d

More generally, for a tensor with multiple indices, each index contributes
a Riemann term.
=#

"""
    commute_covds(expr, covd_name; registry=current_registry()) -> TensorExpr

Commute covariant derivatives, inserting Riemann curvature terms.
Sorts derivatives into canonical (alphabetical) order.
"""
function commute_covds(expr::TensorExpr, covd::Symbol;
                       registry::TensorRegistry=current_registry())
    with_registry(registry) do
        _commute_covds_fixpoint(expr, covd, registry)
    end
end

function _commute_covds_fixpoint(expr::TensorExpr, covd::Symbol, reg::TensorRegistry; maxiter=20)
    current = expr
    for _ in 1:maxiter
        next = _commute_one_pass(current, covd, reg)
        next == current && return current
        current = next
    end
    current
end

function _commute_one_pass(expr::Tensor, ::Symbol, ::TensorRegistry)
    expr
end
function _commute_one_pass(expr::TScalar, ::Symbol, ::TensorRegistry)
    expr
end
function _commute_one_pass(expr::TSum, covd::Symbol, reg::TensorRegistry)
    tsum(TensorExpr[_commute_one_pass(t, covd, reg) for t in expr.terms])
end
function _commute_one_pass(expr::TProduct, covd::Symbol, reg::TensorRegistry)
    TProduct(expr.scalar, TensorExpr[_commute_one_pass(f, covd, reg) for f in expr.factors])
end

function _commute_one_pass(expr::TDeriv, covd::Symbol, reg::TensorRegistry)
    # Look for ∇_a(∇_b(T)) where a > b alphabetically → swap and add Riemann
    inner = _commute_one_pass(expr.arg, covd, reg)

    if inner isa TDeriv
        outer_idx = expr.index
        inner_idx = inner.index

        # If outer derivative index should come after inner, swap them
        if outer_idx.name > inner_idx.name
            # [∇_a, ∇_b] T = ∇_a(∇_b(T)) - ∇_b(∇_a(T))
            # So ∇_a(∇_b(T)) = ∇_b(∇_a(T)) + [∇_a, ∇_b] T
            swapped = TDeriv(inner_idx, TDeriv(outer_idx, inner.arg, expr.covd), expr.covd)
            commutator = _commutator_term(outer_idx, inner_idx, inner.arg, covd, reg)
            return swapped + commutator
        end
    end

    TDeriv(expr.index, inner, expr.covd)
end

"""
    commute_covds(expr, covd, idx_a, idx_b; registry=current_registry()) -> TensorExpr

Commute only the specific pair of covariant derivatives ∇_{idx_a} and ∇_{idx_b}.
"""
function commute_covds(expr::TensorExpr, covd::Symbol,
                       idx_a::Symbol, idx_b::Symbol;
                       registry::TensorRegistry=current_registry())
    with_registry(registry) do
        _commute_specific_pair(expr, covd, idx_a, idx_b, registry)
    end
end

function _commute_specific_pair(expr::TDeriv, covd::Symbol,
                                 idx_a::Symbol, idx_b::Symbol,
                                 reg::TensorRegistry)
    inner = _commute_specific_pair(expr.arg, covd, idx_a, idx_b, reg)

    if inner isa TDeriv
        outer_idx = expr.index
        inner_idx = inner.index
        if outer_idx.name == idx_a && inner_idx.name == idx_b
            swapped = TDeriv(inner_idx, TDeriv(outer_idx, inner.arg, expr.covd), expr.covd)
            commutator = _commutator_term(outer_idx, inner_idx, inner.arg, covd, reg)
            return swapped + commutator
        end
    end
    TDeriv(expr.index, inner, expr.covd)
end

function _commute_specific_pair(expr::TSum, covd::Symbol,
                                 idx_a::Symbol, idx_b::Symbol,
                                 reg::TensorRegistry)
    tsum(TensorExpr[_commute_specific_pair(t, covd, idx_a, idx_b, reg) for t in expr.terms])
end

function _commute_specific_pair(expr::TProduct, covd::Symbol,
                                 idx_a::Symbol, idx_b::Symbol,
                                 reg::TensorRegistry)
    TProduct(expr.scalar, TensorExpr[_commute_specific_pair(f, covd, idx_a, idx_b, reg)
                                      for f in expr.factors])
end

_commute_specific_pair(expr::Tensor, ::Symbol, ::Symbol, ::Symbol, ::TensorRegistry) = expr
_commute_specific_pair(expr::TScalar, ::Symbol, ::Symbol, ::Symbol, ::TensorRegistry) = expr

"""
    sort_covds_to_box(expr::TensorExpr; metric::Symbol=:g) -> TensorExpr

Rewrite derivative expressions to expose the d'Alembertian (box) operator
□ = ∇^a ∇_a = g^{ab} ∂_a ∂_b.

Detects patterns like ∂_a(∂^a(T)) and labels them as box(T).
"""
function sort_covds_to_box(expr::TensorExpr; metric::Symbol=:g)
    walk(expr) do node
        node isa TDeriv || return node
        inner = node.arg
        inner isa TDeriv || return node
        outer_idx = node.index
        inner_idx = inner.index
        # Detect ∂_a(∂^a(T)) or ∂^a(∂_a(T)): same name, opposite positions
        if outer_idx.name == inner_idx.name && outer_idx.position != inner_idx.position
            # Rewrite as g^{ab}∂_a∂_b T (box operator structure)
            # Introduce explicit metric contraction
            used = Set{Symbol}()
            for idx in indices(inner.arg)
                push!(used, idx.name)
            end
            a = fresh_index(used); push!(used, a)
            b = fresh_index(used)
            # □T = g^{ab} ∂_a(∂_b(T))
            return tproduct(1 // 1, TensorExpr[
                Tensor(metric, [up(a), up(b)]),
                TDeriv(down(a), TDeriv(down(b), inner.arg, inner.covd), node.covd)
            ])
        end
        node
    end
end

"""
    sort_covds_to_div(expr::TensorExpr) -> TensorExpr

Rewrite derivative expressions to expose divergence patterns.
Detects ∂_a T^{a...} where a derivative index contracts with an index of
the argument tensor. Returns the expression unchanged (pattern detection only;
the contraction is already canonical).
"""
function sort_covds_to_div(expr::TensorExpr)
    # Divergence patterns (∂_a T^{a...}) are already represented naturally
    # in the AST as TDeriv(down(:a), Tensor(:T, [up(:a), ...])).
    # No transformation needed — the pattern is already exposed.
    expr
end

"""
    symmetrize_covds(expr, covd_name; registry=current_registry()) -> TensorExpr

Express double covariant derivatives in symmetrized form.

For **scalars** (no free indices on the differentiated object):
  ∇_a ∇_b φ is already symmetric, returned unchanged.

For **tensors** with indices:
  ∇_a ∇_b T^c = ∇_{(a} ∇_{b)} T^c + ½[∇_a, ∇_b] T^c

where the symmetrized part is ∇_{(a} ∇_{b)} T^c = ½(∇_a ∇_b T^c + ∇_b ∇_a T^c)
and the commutator produces Riemann curvature terms:
  [∇_a, ∇_b] V^c = R^c_{dab} V^d

Only derivative pairs matching `covd_name` are symmetrized; other derivatives
pass through unchanged.
"""
function symmetrize_covds(expr::TensorExpr, covd::Symbol;
                          registry::TensorRegistry=current_registry())
    with_registry(registry) do
        _symmetrize_covds_walk(expr, covd, registry)
    end
end

function _symmetrize_covds_walk(expr::TDeriv, covd::Symbol, reg::TensorRegistry)
    inner = _symmetrize_covds_walk(expr.arg, covd, reg)

    if inner isa TDeriv
        outer_idx = expr.index
        inner_idx = inner.index
        body = inner.arg

        # Scalar case: no indices on body means derivatives commute -- skip
        if _is_scalar_body(body)
            return TDeriv(expr.index, inner, expr.covd)
        end

        # Tensor case: ∇_a(∇_b(T)) = ½(∇_a∇_b + ∇_b∇_a)(T) + ½[∇_a,∇_b](T)
        original = TDeriv(outer_idx, TDeriv(inner_idx, body, inner.covd), expr.covd)
        swapped  = TDeriv(inner_idx, TDeriv(outer_idx, body, expr.covd), inner.covd)
        sym_part = (1 // 2) * (original + swapped)
        comm_part = (1 // 2) * _commutator_term(outer_idx, inner_idx, body, covd, reg)
        return sym_part + comm_part
    end
    TDeriv(expr.index, inner, expr.covd)
end

function _symmetrize_covds_walk(expr::TSum, covd::Symbol, reg::TensorRegistry)
    tsum(TensorExpr[_symmetrize_covds_walk(t, covd, reg) for t in expr.terms])
end
function _symmetrize_covds_walk(expr::TProduct, covd::Symbol, reg::TensorRegistry)
    TProduct(expr.scalar, TensorExpr[_symmetrize_covds_walk(f, covd, reg) for f in expr.factors])
end
_symmetrize_covds_walk(expr::Tensor, ::Symbol, ::TensorRegistry) = expr
_symmetrize_covds_walk(expr::TScalar, ::Symbol, ::TensorRegistry) = expr

"""Check whether a tensor expression is a scalar (has no free indices on its own)."""
function _is_scalar_body(expr::TensorExpr)
    expr isa TScalar && return true
    expr isa Tensor && return isempty(expr.indices)
    # For sums/products/derivs, check free indices
    isempty(free_indices(expr))
end

"""
Compute [∇_a, ∇_b] T for a tensor T.
For each index on T, adds a Riemann curvature term.
"""
function _commutator_term(a::TIndex, b::TIndex, tensor::TensorExpr,
                           covd::Symbol, reg::TensorRegistry)
    tensor isa Tensor || return ZERO  # only handle bare tensors for now

    used = Set{Symbol}()
    push!(used, a.name, b.name)
    for idx in tensor.indices
        push!(used, idx.name)
    end

    result = ZERO

    for (i, tidx) in enumerate(tensor.indices)
        dummy = fresh_index(used)
        push!(used, dummy)

        new_indices = copy(tensor.indices)

        if tidx.position == Up
            # +R^{c}_{dab} T^{...d...}
            riem = Tensor(:Riem, [tidx, down(dummy), a, b])
            new_indices[i] = up(dummy)
            t_mod = Tensor(tensor.name, new_indices)
            result = result + riem * t_mod
        else
            # -R^{d}_{cab} T^{...d...}
            riem = Tensor(:Riem, [up(dummy), tidx, a, b])
            new_indices[i] = down(dummy)
            t_mod = Tensor(tensor.name, new_indices)
            result = result - riem * t_mod
        end
    end

    result
end
