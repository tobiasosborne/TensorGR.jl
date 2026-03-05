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
