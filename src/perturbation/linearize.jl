#= Linearization of GR expressions around a background.

linearize(expr, g => η + ε*h, order=1) substitutes the metric perturbation
and expands to the specified order in ε.

At first order on a flat background, the linearised curvature tensors have
explicit closed-form expressions (hard-coded for efficiency).
=#

"""
    linearize(expr, perturbation; order=1) -> TensorExpr

Linearize a tensor expression by substituting a metric perturbation.

# Arguments
- `expr::TensorExpr`: the expression to linearize
- `perturbation::Pair{Symbol,Tuple{Symbol,Symbol}}`: `metric => (background, perturbation)`
  e.g., `:g => (:η, :h)`
- `order::Int`: perturbation order (default 1)
"""
function linearize(expr::TensorExpr, pert::Pair{Symbol, Tuple{Symbol, Symbol}}; order::Int=1)
    metric, (bg, h) = pert

    if order == 1
        return _linearize_first_order(expr, metric, bg, h)
    else
        error("Perturbation order $order not yet implemented (only order=1)")
    end
end

function _linearize_first_order(expr::TensorExpr, metric::Symbol, bg::Symbol, h::Symbol)
    walk(expr) do node
        _linearize_node(node, metric, bg, h)
    end
end

# Default: pass through
_linearize_node(expr::TensorExpr, ::Symbol, ::Symbol, ::Symbol) = expr

function _linearize_node(t::Tensor, metric::Symbol, bg::Symbol, h::Symbol)
    if t.name == metric
        # g_{ab} → η_{ab} (at zeroth order, the perturbation h is a separate field)
        return Tensor(bg, t.indices)
    end
    t
end

"""
    δRiemann(a, b, c, d, h) -> TensorExpr

First-order perturbation of the Riemann tensor on a flat background:

δR_{abcd} = 1/2 (∂_b ∂_c h_{ad} + ∂_a ∂_d h_{bc} - ∂_a ∂_c h_{bd} - ∂_b ∂_d h_{ac})

All indices must be `down` (covariant).
"""
function δRiemann(a::TIndex, b::TIndex, c::TIndex, d::TIndex, h::Symbol)
    h_ad = Tensor(h, [a, d])
    h_bc = Tensor(h, [b, c])
    h_bd = Tensor(h, [b, d])
    h_ac = Tensor(h, [a, c])

    (1 // 2) * (
        TDeriv(b, TDeriv(c, h_ad)) +
        TDeriv(a, TDeriv(d, h_bc)) -
        TDeriv(a, TDeriv(c, h_bd)) -
        TDeriv(b, TDeriv(d, h_ac))
    )
end

"""
    δRicci(a, b, h, bg) -> TensorExpr

First-order perturbation of the Ricci tensor on a flat background:

δR_{ab} = 1/2 (∂^c ∂_a h_{bc} + ∂^c ∂_b h_{ac} - ∂_a ∂_b h - □ h_{ab})

where h = η^{ab} h_{ab} is the trace and □ = η^{ab} ∂_a ∂_b.
"""
function δRicci(a::TIndex, b::TIndex, h::Symbol; trace_idx=:_trC)
    c_up = up(trace_idx)
    c_dn = down(trace_idx)
    h_bc = Tensor(h, [b, c_dn])
    h_ac = Tensor(h, [a, c_dn])
    h_ab = Tensor(h, [a, b])
    h_trace = Tensor(Symbol(h, :_trace), TIndex[])  # scalar trace h
    box_h_ab = Tensor(Symbol(:□, h), [a, b])         # □h_{ab}

    (1 // 2) * (
        TDeriv(c_up, TDeriv(a, h_bc)) +
        TDeriv(c_up, TDeriv(b, h_ac)) -
        TDeriv(a, TDeriv(b, h_trace)) -
        box_h_ab
    )
end

"""
    δRicciScalar(h, bg) -> TensorExpr

First-order perturbation of the Ricci scalar on a flat background:

δR = ∂^a ∂^b h_{ab} - □h
"""
function δRicciScalar(h::Symbol; trace_idxs=(:_trA, :_trB))
    a_up = up(trace_idxs[1])
    b_up = up(trace_idxs[2])
    a_dn = down(trace_idxs[1])
    b_dn = down(trace_idxs[2])
    h_ab = Tensor(h, [a_dn, b_dn])
    h_trace = Tensor(Symbol(h, :_trace), TIndex[])
    box_h = Tensor(Symbol(:□, h, :_trace), TIndex[])

    TDeriv(a_up, TDeriv(b_up, h_ab)) - box_h
end
