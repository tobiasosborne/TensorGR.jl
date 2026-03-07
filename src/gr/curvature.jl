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

    # Register standard LaTeX aliases for the tex"..." parser
    tex_alias!(reg, :R, :Riem; rank=4)
    tex_alias!(reg, :R, :Ric; rank=2)
    tex_alias!(reg, :R, :RicScalar; rank=0)
    tex_alias!(reg, :G, :Ein; rank=2)
    tex_alias!(reg, :C, :Weyl; rank=4)
    tex_alias!(reg, :S, :Sch; rank=2)

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

"""
    cotton_expr(a, b, metric; epsilon=:εg, dim=3) -> TensorExpr

Construct the Cotton tensor in 3D:
`C_{ab} = ε_a^{cd} ∇_c (R_{db} - R/4 g_{db})`

The Cotton tensor is the 3D analogue of the Weyl tensor (Weyl = 0 in 3D).
A 3D space is conformally flat iff its Cotton tensor vanishes.
"""
function cotton_expr(a::TIndex, b::TIndex, metric::Symbol;
                      epsilon::Symbol=Symbol(:ε, metric), dim::Int=3)
    dim == 3 || error("Cotton tensor is only defined in 3 dimensions, got dim=$dim")

    used = Set{Symbol}([a.name, b.name])
    c = fresh_index(used); push!(used, c)
    d = fresh_index(used); push!(used, d)

    # ε_a^{cd}
    eps = Tensor(epsilon, [a, up(c), up(d)])
    # ∇_c(R_{db} - R/4 g_{db}) — represented as TDeriv(c, ...)
    schouten_db = Tensor(:Ric, [down(d), b]) -
                   (1 // (2 * (dim - 1))) * Tensor(:RicScalar, TIndex[]) * Tensor(metric, [down(d), b])
    eps * TDeriv(down(c), schouten_db)
end

"""
    tensor_norm(t::Tensor, metric::Symbol; registry=current_registry()) -> TensorExpr

Contract a tensor with itself using the metric:
`‖T‖² = T_{a₁...aₙ} T^{a₁...aₙ}`

All indices are assumed down; raising is done with the inverse metric.
"""
function tensor_norm(t::Tensor, metric::Symbol;
                      registry::TensorRegistry=current_registry())
    with_registry(registry) do
        n = length(t.indices)
        n == 0 && return t * t  # scalar: just square

        used = Set{Symbol}(idx.name for idx in t.indices)
        raised_indices = TIndex[]
        for idx in t.indices
            fresh = fresh_index(used)
            push!(used, fresh)
            push!(raised_indices, up(fresh))
        end
        t_up = Tensor(t.name, raised_indices)
        t * t_up
    end
end
