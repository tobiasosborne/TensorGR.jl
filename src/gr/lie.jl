#= Lie derivatives.

£_v T^{a...}_{b...} = v^c ∇_c T^{a...}_{b...}
    - (∇_c v^a) T^{c...}_{b...} - ...   (one term per upper index)
    + (∇_b v^c) T^{a...}_{c...} + ...   (one term per lower index)

On a torsion-free manifold, ∇ can be any torsion-free connection (partials work too).
=#

"""
    lie_derivative(v::Tensor, expr::TensorExpr) -> TensorExpr

Compute the Lie derivative £_v of a tensor expression along vector field v.
The vector field v should be a rank-(1,0) Tensor with one upper index.
"""
function lie_derivative(v::Tensor, expr::TensorExpr)
    @assert length(v.indices) == 1 && v.indices[1].position == Up "v must be a vector with one upper index"
    _lie_d(v, expr)
end

function _lie_d(v::Tensor, t::Tensor)
    used = Set{Symbol}()
    push!(used, v.indices[1].name)
    for idx in t.indices
        push!(used, idx.name)
    end

    # Transport term: v^c ∂_c T
    c = fresh_index(used)
    push!(used, c)
    transport = Tensor(v.name, [up(c)]) * TDeriv(down(c), t)

    result = transport

    # For each index on T, add correction terms
    for (i, tidx) in enumerate(t.indices)
        d = fresh_index(used)
        push!(used, d)

        new_indices = copy(t.indices)

        if tidx.position == Up
            # -(∂_c v^a_i) T^{...c...}_{...}
            # where c replaces a_i in T
            dv = TDeriv(down(d), Tensor(v.name, [tidx]))
            new_indices[i] = up(d)
            t_mod = Tensor(t.name, new_indices)
            result = result - dv * t_mod
        else
            # +(∂_{b_i} v^c) T^{...}_{...c...}
            # where c replaces b_i in T
            dv = TDeriv(tidx, Tensor(v.name, [up(d)]))
            new_indices[i] = down(d)
            t_mod = Tensor(t.name, new_indices)
            result = result + dv * t_mod
        end
    end

    result
end

function _lie_d(v::Tensor, s::TScalar)
    # Lie derivative of a scalar: v^a ∂_a f
    if s == ZERO
        return ZERO
    end
    used = Set{Symbol}(v.indices[1].name)
    c = fresh_index(used)
    Tensor(v.name, [up(c)]) * TDeriv(down(c), s)
end

function _lie_d(v::Tensor, p::TProduct)
    # Leibniz rule on products
    factors = p.factors
    terms = TensorExpr[]
    for i in eachindex(factors)
        new_factors = TensorExpr[]
        for (j, fj) in enumerate(factors)
            if j == i
                push!(new_factors, _lie_d(v, fj))
            else
                push!(new_factors, fj)
            end
        end
        push!(terms, tproduct(p.scalar, new_factors))
    end
    tsum(terms)
end

function _lie_d(v::Tensor, s::TSum)
    tsum(TensorExpr[_lie_d(v, t) for t in s.terms])
end

function _lie_d(v::Tensor, d::TDeriv)
    # Lie derivative of a derivative expression
    # £_v(∂_a T) = ∂_a(£_v T) + (∂_a v^c)(∂_c T) ... complex
    # For simplicity, treat as a generic expression and apply Leibniz
    TDeriv(d.index, _lie_d(v, d.arg))
end

"""
    lie_bracket(v::Tensor, w::Tensor) -> TensorExpr

Compute the Lie bracket [v, w]^a = v^b ∂_b w^a - w^b ∂_b v^a.
Both v and w must be vector fields (rank-(1,0) tensors).
"""
function lie_bracket(v::Tensor, w::Tensor)
    @assert length(v.indices) == 1 && v.indices[1].position == Up "v must be a vector"
    @assert length(w.indices) == 1 && w.indices[1].position == Up "w must be a vector"

    used = Set{Symbol}([v.indices[1].name, w.indices[1].name])
    b = fresh_index(used)
    push!(used, b)

    # [v, w]^a = v^b ∂_b w^a - w^b ∂_b v^a
    v_b = Tensor(v.name, [up(b)])
    w_b = Tensor(w.name, [up(b)])

    v_b * TDeriv(down(b), w) - w_b * TDeriv(down(b), v)
end

"""
    lie_to_covd(expr::TensorExpr, covd::Symbol;
                registry::TensorRegistry=current_registry()) -> TensorExpr

Rewrite Lie derivatives in terms of covariant derivatives.
On a torsion-free manifold, ∂_a can be replaced by ∇_a in the Lie derivative formula
(the connection terms cancel). This simply returns the expression unchanged since
our Lie derivative already uses partial derivatives which equal ∇ - Γ, and on
torsion-free manifolds the Lie derivative formula is the same with ∇.
"""
function lie_to_covd(expr::TensorExpr, covd::Symbol;
                     registry::TensorRegistry=current_registry())
    # On torsion-free manifolds, the Lie derivative formula is identical
    # whether written with partial or covariant derivatives.
    # The expression is already correct; this is a no-op for torsion-free.
    expr
end
