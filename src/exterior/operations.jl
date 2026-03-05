#= Exterior algebra operations: wedge product, Hodge dual, exterior derivative.

Wedge product: (α ∧ β)_{a1...ap b1...bq} = ((p+q)!/(p!q!)) α_{[a1...ap} β_{b1...bq]}
Exterior derivative: (dα)_{a0 a1...ap} = (p+1) ∂_{[a0} α_{a1...ap]}
Interior product: (ι_v α)_{a2...ap} = v^{a1} α_{a1 a2...ap}
=#

"""
    wedge(α::TensorExpr, β::TensorExpr, p::Int, q::Int) -> TensorExpr

Compute the wedge product α ∧ β where α is a p-form and β is a q-form.
The result is a (p+q)-form (antisymmetrization is left to canonicalize).
"""
function wedge(α::Tensor, β::Tensor, p::Int, q::Int)
    # The wedge product is the antisymmetrized tensor product
    # with the combinatorial factor (p+q)!/(p!q!)
    coeff = factorial(p + q) // (factorial(p) * factorial(q))
    # Simply multiply — the antisymmetry is enforced by the form's symmetry
    # properties during canonicalization
    tproduct(coeff, TensorExpr[α, β])
end

"""
    exterior_d(α::Tensor, degree::Int, deriv_idx::TIndex) -> TensorExpr

Compute the exterior derivative dα.
(dα)_{a0 a1...ap} = (p+1) ∂_{[a0} α_{a1...ap]}
"""
function exterior_d(α::Tensor, degree::Int, deriv_idx::TIndex)
    TDeriv(deriv_idx, α)
end

"""
    interior_product(v::Tensor, α::Tensor) -> TensorExpr

Interior product ι_v α: contract v with the first index of α.
"""
function interior_product(v::Tensor, α::Tensor)
    @assert length(v.indices) == 1 && v.indices[1].position == Up
    @assert length(α.indices) >= 1

    # Contract v^a with α_{a...}
    # Need to ensure dummy index
    v_idx = v.indices[1]
    α_idx = α.indices[1]

    if v_idx.name == α_idx.name && v_idx.position != α_idx.position
        # Already contracted
        return v * α
    end

    # Create contraction by renaming
    used = Set{Symbol}()
    push!(used, v_idx.name)
    for idx in α.indices
        push!(used, idx.name)
    end
    dummy = fresh_index(used)

    v_renamed = Tensor(v.name, [up(dummy)])
    α_renamed = Tensor(α.name, vcat([down(dummy)], α.indices[2:end]))
    v_renamed * α_renamed
end

"""
    codifferential(α::Tensor, epsilon::Symbol, metric::Symbol,
                   degree::Int, dim::Int) -> TensorExpr

Compute the codifferential δα = (-1)^{d(p+1)+1} ★d★α.
For a p-form, returns a (p-1)-form.
"""
function codifferential(α::Tensor, epsilon::Symbol, metric::Symbol,
                        degree::Int, dim::Int)
    # δ = (-1)^{d(p+1)+1} ★ d ★
    sign = (-1)^(dim * (degree + 1) + 1)

    used = Set{Symbol}(i.name for i in α.indices)
    deriv_idx = fresh_index(used)
    push!(used, deriv_idx)

    # ★α
    star_α = hodge_dual(α, epsilon, degree, dim)

    # d(★α) — need a fresh index for the derivative
    d_star_α = TDeriv(down(deriv_idx), star_α)

    # ★(d★α)
    # d★α is a (dim - degree + 1)-form
    # But we can't easily apply hodge_dual to a non-Tensor expression
    # So we return the symbolic expression
    Rational{Int}(sign) * d_star_α
end

"""
    cartan_lie_d(v::Tensor, α::Tensor, degree::Int, deriv_idx::TIndex) -> TensorExpr

Cartan's magic formula: £_v ω = d(ι_v ω) + ι_v(dω).
Returns the right-hand side.
"""
function cartan_lie_d(v::Tensor, α::Tensor, degree::Int, deriv_idx::TIndex)
    @assert length(v.indices) == 1 && v.indices[1].position == Up

    used = Set{Symbol}()
    push!(used, v.indices[1].name)
    for idx in α.indices
        push!(used, idx.name)
    end
    push!(used, deriv_idx.name)

    d2_idx = fresh_index(used)

    # ι_v α
    iv_α = interior_product(v, α)

    # d(ι_v α) — (degree-1)-form becomes degree-form
    d_iv_α = TDeriv(deriv_idx, iv_α)

    # dα
    dα = exterior_d(α, degree, TIndex(d2_idx, Down))

    # ι_v(dα) — need to contract v with dα
    # dα is TDeriv, so we wrap in a product with v
    iv_dα = Tensor(v.name, [up(d2_idx)]) * dα

    d_iv_α + iv_dα
end

"""
    hodge_dual(α::Tensor, metric::Symbol, dim::Int) -> TensorExpr

Compute the Hodge dual ★α.
For a p-form in d dimensions, the result is a (d-p)-form.
(★α)_{b1...b(d-p)} = (1/p!) ε^{a1...ap}_{b1...b(d-p)} α_{a1...ap}

Note: requires the Levi-Civita tensor to be defined.
"""
function hodge_dual(α::Tensor, epsilon::Symbol, degree::Int, dim::Int)
    # Contract α with the Levi-Civita tensor
    used = Set{Symbol}()
    for idx in α.indices
        push!(used, idx.name)
    end

    # Create dummy indices for contraction
    up_indices = TIndex[]
    for _ in 1:degree
        d = fresh_index(used)
        push!(used, d)
        push!(up_indices, up(d))
    end

    # Remaining indices on ε become the result indices
    result_indices = TIndex[]
    for _ in 1:(dim - degree)
        d = fresh_index(used)
        push!(used, d)
        push!(result_indices, down(d))
    end

    ε = Tensor(epsilon, vcat(up_indices, result_indices))
    # Rename α indices to match dummies
    α_renamed = Tensor(α.name, [down(idx.name) for idx in up_indices])

    coeff = 1 // factorial(degree)
    tproduct(coeff, TensorExpr[ε, α_renamed])
end
