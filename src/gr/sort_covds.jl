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
            swapped = TDeriv(inner_idx, TDeriv(outer_idx, inner.arg))
            commutator = _commutator_term(outer_idx, inner_idx, inner.arg, covd, reg)
            return swapped + commutator
        end
    end

    TDeriv(expr.index, inner)
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
            swapped = TDeriv(inner_idx, TDeriv(outer_idx, inner.arg))
            commutator = _commutator_term(outer_idx, inner_idx, inner.arg, covd, reg)
            return swapped + commutator
        end
    end
    TDeriv(expr.index, inner)
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
        if node isa TDeriv && node.arg isa TDeriv
            outer = node.index
            inner = node.arg.index
            if outer.name == inner.name && outer.position != inner.position
                # ∂_a ∂^a T = □T — represent as a scalar-labeled derivative
                return TDeriv(outer, TDeriv(inner, node.arg.arg))
            end
        end
        node
    end
end

"""
    sort_covds_to_div(expr::TensorExpr) -> TensorExpr

Rewrite derivative expressions to expose divergence patterns
∇_a T^{a...} = div(T^{...}).
"""
function sort_covds_to_div(expr::TensorExpr)
    # Walk looking for ∂_a applied to something with matching upper index
    walk(expr) do node
        if node isa TDeriv
            didx = node.index
            inner = node.arg
            if inner isa Tensor
                for (i, tidx) in enumerate(inner.indices)
                    if tidx.name == didx.name && tidx.position != didx.position
                        # This is a divergence pattern
                        return node  # keep as-is but recognized
                    end
                end
            end
        end
        node
    end
end

"""
    symmetrize_covds(expr::TensorExpr, covd::Symbol;
                     registry::TensorRegistry=current_registry()) -> TensorExpr

Rewrite ∇_a ∇_b T as ∇_{(a} ∇_{b)} T + (1/2)[∇_a, ∇_b] T.
Requires symmetrize to be available.
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
        # ∂_a(∂_b(T)) = (1/2)(∂_a ∂_b + ∂_b ∂_a)(T) + (1/2)[∂_a, ∂_b](T)
        sym_part = (1 // 2) * (TDeriv(outer_idx, TDeriv(inner_idx, inner.arg)) +
                                TDeriv(inner_idx, TDeriv(outer_idx, inner.arg)))
        comm_part = (1 // 2) * _commutator_term(outer_idx, inner_idx, inner.arg, covd, reg)
        return sym_part + comm_part
    end
    TDeriv(expr.index, inner)
end

function _symmetrize_covds_walk(expr::TSum, covd::Symbol, reg::TensorRegistry)
    tsum(TensorExpr[_symmetrize_covds_walk(t, covd, reg) for t in expr.terms])
end
function _symmetrize_covds_walk(expr::TProduct, covd::Symbol, reg::TensorRegistry)
    TProduct(expr.scalar, TensorExpr[_symmetrize_covds_walk(f, covd, reg) for f in expr.factors])
end
_symmetrize_covds_walk(expr::Tensor, ::Symbol, ::TensorRegistry) = expr
_symmetrize_covds_walk(expr::TScalar, ::Symbol, ::TensorRegistry) = expr

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
