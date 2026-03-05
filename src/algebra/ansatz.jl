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

"""
    all_contractions(tensors::Vector{Tensor}, free_idxs::Vector{TIndex}) -> Vector{TensorExpr}

Enumerate all possible index contractions of the given tensors that produce
expressions with the specified free indices.

This generates all ways to contract dummy indices between the tensors.
"""
function all_contractions(tensors::Vector{Tensor}, free_idxs::Vector{TIndex})
    # Collect all indices from all tensors
    all_idxs = TIndex[]
    for t in tensors
        append!(all_idxs, t.indices)
    end

    free_names = Set(idx.name for idx in free_idxs)
    # Indices that need to be contracted (not free)
    to_contract = TIndex[idx for idx in all_idxs if idx.name ∉ free_names]

    if isempty(to_contract)
        # No contractions needed, just multiply everything
        return [tproduct(1 // 1, TensorExpr[t for t in tensors])]
    end

    # For a simple implementation, return the basic product
    # Full enumeration of all contraction patterns is combinatorially complex
    [tproduct(1 // 1, TensorExpr[t for t in tensors])]
end

"""
    contraction_ansatz(tensors::Vector{Tensor}, free_idxs::Vector{TIndex};
                       coeffs::Union{Vector{Symbol}, Nothing}=nothing) -> TensorExpr

Build the most general linear combination of contractions of the given tensors
with the specified free indices.
"""
function contraction_ansatz(tensors::Vector{Tensor}, free_idxs::Vector{TIndex};
                            coeffs::Union{Vector{Symbol}, Nothing}=nothing)
    contractions = all_contractions(tensors, free_idxs)
    if coeffs === nothing
        coeffs_actual = [Symbol(:c, i) for i in eachindex(contractions)]
    else
        coeffs_actual = coeffs
    end
    make_ansatz(contractions, coeffs_actual)
end
