#= GR curvature tensor definitions.

These functions register the standard GR tensors (Riemann, Ricci, Einstein,
Weyl, Schouten) and define their relationships (trace, decomposition).
=#

"""
    define_curvature_tensors!(reg, manifold, metric)

Register the standard curvature tensors for the given manifold and metric.
"""
function define_curvature_tensors!(reg::TensorRegistry, manifold::Symbol, metric::Symbol)
    mp = get_manifold(reg, manifold)
    d = mp.dim

    # Kronecker delta
    has_tensor(reg, :δ) || register_tensor!(reg, TensorProperties(
        name=:δ, manifold=manifold, rank=(1, 1), symmetries=Any[],
        options=Dict{Symbol,Any}(:is_delta => true)))

    # Riemann tensor R_{abcd}
    register_tensor!(reg, TensorProperties(
        name=:Riem, manifold=manifold, rank=(0, 4),
        symmetries=Any[RiemannSymmetry()],
        options=Dict{Symbol,Any}()))

    # Ricci tensor R_{ab} = R^c_{acb}
    register_tensor!(reg, TensorProperties(
        name=:Ric, manifold=manifold, rank=(0, 2),
        symmetries=Any[Symmetric(1, 2)],
        options=Dict{Symbol,Any}()))

    # Ricci scalar R = g^{ab} R_{ab}
    register_tensor!(reg, TensorProperties(
        name=:RicScalar, manifold=manifold, rank=(0, 0),
        symmetries=Any[],
        options=Dict{Symbol,Any}()))

    # Einstein tensor G_{ab} = R_{ab} - (1/2) g_{ab} R
    register_tensor!(reg, TensorProperties(
        name=:Ein, manifold=manifold, rank=(0, 2),
        symmetries=Any[Symmetric(1, 2)],
        options=Dict{Symbol,Any}()))

    # Weyl tensor C_{abcd}
    register_tensor!(reg, TensorProperties(
        name=:Weyl, manifold=manifold, rank=(0, 4),
        symmetries=Any[RiemannSymmetry()],
        options=Dict{Symbol,Any}(:traceless => true)))

    # Schouten tensor S_{ab}
    register_tensor!(reg, TensorProperties(
        name=:Sch, manifold=manifold, rank=(0, 2),
        symmetries=Any[Symmetric(1, 2)],
        options=Dict{Symbol,Any}()))

    nothing
end

"""
    ricci_from_riemann(Riem_expr, metric_name, idx_a, idx_c)

Contract the Riemann tensor to form the Ricci tensor:
R_{ac} = g^{bd} R_{abcd} (trace over 1st and 3rd indices).

Returns the expression with metric contraction ready to apply.
"""
function ricci_from_riemann(idx_a::TIndex, idx_c::TIndex; trace_idx=:_b)
    b_up = up(trace_idx)
    b_dn = down(trace_idx)
    # R^b_{abc} = g^{bd} R_{dabc} ... actually simpler:
    # Ric_{ac} = Riem^{b}_{abc} (standard convention: trace on 1st & 3rd)
    Tensor(:Riem, [b_up, idx_a, b_dn, idx_c])
end

"""
    einstein_tensor(idx_a, idx_b, metric_name)

G_{ab} = Ric_{ab} - (1/2) g_{ab} RicScalar
"""
function einstein_expr(idx_a::TIndex, idx_b::TIndex, metric::Symbol)
    Tensor(:Ric, [idx_a, idx_b]) -
        (1 // 2) * Tensor(metric, [idx_a, idx_b]) * Tensor(:RicScalar, TIndex[])
end
