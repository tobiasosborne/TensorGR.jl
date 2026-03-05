#= Projection operators for SVT decomposition.

In Fourier space with spatial momentum k_i:

Transverse projector:
  P^T_{ij}(k) = δ_{ij} - k_i k_j / k²

TT projector (3D, symmetric rank-2):
  Π^TT_{ijkl} = 1/2 (P^T_{ik} P^T_{jl} + P^T_{il} P^T_{jk} - P^T_{ij} P^T_{kl})

These are represented as symbolic TensorExpr nodes.
=#

"""
    transverse_projector(i, j, k_name, k_sq_name) -> TensorExpr

P^T_{ij} = δ_{ij} - k_i k_j / k²
"""
function transverse_projector(i::TIndex, j::TIndex;
                              k_name::Symbol=:k, k_sq::Symbol=:k²)
    δ_ij = Tensor(:δ, [i, j])
    k_i = Tensor(k_name, [i])
    k_j = Tensor(k_name, [j])
    k_sq_scalar = TScalar(k_sq)

    # δ_{ij} - k_i k_j / k²
    # We represent 1/k² as TScalar(:(1/k²)) for now
    δ_ij - TProduct(1 // 1, TensorExpr[TScalar(:(1 / $k_sq)), k_i, k_j])
end

"""
    tt_projector(i, j, k, l; kwargs...) -> TensorExpr

Π^TT_{ijkl} = 1/2 (P^T_{ik} P^T_{jl} + P^T_{il} P^T_{jk} - P^T_{ij} P^T_{kl})
"""
function tt_projector(i::TIndex, j::TIndex, k::TIndex, l::TIndex; kwargs...)
    P_ik = transverse_projector(i, k; kwargs...)
    P_jl = transverse_projector(j, l; kwargs...)
    P_il = transverse_projector(i, l; kwargs...)
    P_jk = transverse_projector(j, k; kwargs...)
    P_ij = transverse_projector(i, j; kwargs...)
    P_kl = transverse_projector(k, l; kwargs...)

    (1 // 2) * (P_ik * P_jl + P_il * P_jk - P_ij * P_kl)
end
