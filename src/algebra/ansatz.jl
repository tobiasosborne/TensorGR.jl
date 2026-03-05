#= Ansatz construction.

Utilities for constructing the most general tensor expression with given
index structure and symmetry properties.
=#

"""
    make_ansatz(terms::Vector{TensorExpr}, coeffs::Vector{Symbol}) -> TensorExpr

Create a linear combination: c₁ T₁ + c₂ T₂ + ...
where cᵢ are symbolic scalar coefficients.
"""
function make_ansatz(terms::Vector{TensorExpr}, coeffs::Vector{Symbol})
    @assert length(terms) == length(coeffs)
    result = TensorExpr[]
    for (t, c) in zip(terms, coeffs)
        push!(result, TScalar(c) * t)
    end
    tsum(result)
end

"""
    make_ansatz(terms::Vector{TensorExpr}) -> TensorExpr

Create a linear combination with auto-generated coefficient names c1, c2, ...
"""
function make_ansatz(terms::Vector{TensorExpr})
    coeffs = [Symbol(:c, i) for i in eachindex(terms)]
    make_ansatz(terms, coeffs)
end
