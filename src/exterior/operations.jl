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
