#= Spin projectors for vector fields A_μ.
#
# A vector field in d dimensions decomposes as A_μ = A_μ^T + A_μ^L where:
#   A_μ^T = θ_{μν} A^ν  (transverse, spin-1, d-1 DOF)
#   A_μ^L = ω_{μν} A^ν  (longitudinal, spin-0, 1 DOF)
#
# The projectors are:
#   P^(1)_{μν} = θ_{μν} = η_{μν} - k_μ k_ν / k²
#   P^(0)_{μν} = ω_{μν} = k_μ k_ν / k²
#
# Properties:
#   Completeness: P^(1) + P^(0) = η_{μν}
#   Idempotency:  P^(i) P^(i) = P^(i)
#   Orthogonality: P^(1) P^(0) = 0
#
# Ground truth: PSALTer arXiv:2406.09500, Sec 3.1.
=#

"""
    vector_spin1_projector(μ::TIndex, ν::TIndex; kwargs...) -> TensorExpr

Spin-1 (transverse) projector for a vector field:

    P^(1)_{μν} = θ_{μν} = η_{μν} - k_μ k_ν / k²

This projects onto the (d-1)-dimensional transverse subspace.
Satisfies k^μ P^(1)_{μν} = 0.

Keyword arguments are passed to `theta_projector`:
- `metric::Symbol=:g` -- background metric name
- `k_name::Symbol=:k` -- momentum tensor name
- `k_sq=:k²`          -- k² symbol for denominator
"""
vector_spin1_projector(μ::TIndex, ν::TIndex; kwargs...) =
    theta_projector(μ, ν; kwargs...)

"""
    vector_spin0_projector(μ::TIndex, ν::TIndex; kwargs...) -> TensorExpr

Spin-0 (longitudinal) projector for a vector field:

    P^(0)_{μν} = ω_{μν} = k_μ k_ν / k²

This projects onto the 1-dimensional longitudinal subspace.
Satisfies P^(0)_{μν} = k_μ k_ν / k².

Keyword arguments are passed to `omega_projector`:
- `k_name::Symbol=:k` -- momentum tensor name
- `k_sq=:k²`          -- k² symbol for denominator
"""
vector_spin0_projector(μ::TIndex, ν::TIndex; kwargs...) =
    omega_projector(μ, ν; _omega_kwargs(kwargs)...)

"""
    vector_spin_project(kernel::TensorExpr, spin::Symbol;
                        idx_left::TIndex=down(:a), idx_right::TIndex=down(:b),
                        kwargs...) -> TensorExpr

Project a rank-2 vector field kernel K_{μν} onto a spin sector.

`spin` must be `:spin1` or `:spin0`.

Returns P^(s)_{μ}^{ρ} K_{ρσ} P^(s)^{σ}_{ν} (double projection).
For a diagonal kernel K_{μν} = f(k²) η_{μν}, this simplifies to
f(k²) P^(s)_{μν} with the appropriate trace factor.

Ground truth: PSALTer arXiv:2406.09500, Sec 3.1.
"""
function vector_spin_project(kernel::TensorExpr, spin::Symbol;
                             idx_left::TIndex=down(:a), idx_right::TIndex=down(:b),
                             kwargs...)
    spin in (:spin1, :spin0) ||
        error("Vector spin must be :spin1 or :spin0, got :$spin")

    used = Set{Symbol}([idx_left.name, idx_right.name])
    rho = fresh_index(used); push!(used, rho)
    sigma = fresh_index(used)

    proj_fn = spin == :spin1 ? vector_spin1_projector : vector_spin0_projector

    P_left = proj_fn(idx_left, up(rho); kwargs...)
    P_right = proj_fn(up(sigma), idx_right; kwargs...)

    # K_{ρσ} with renamed indices for contraction
    K_inner = rename_dummy(kernel, idx_left.name, rho)
    K_inner = rename_dummy(K_inner, idx_right.name, sigma)

    tproduct(1 // 1, TensorExpr[P_left, K_inner, P_right])
end
