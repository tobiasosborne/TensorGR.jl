#= Spin projectors for antisymmetric rank-2 field B_{[μν]}.
#
# An antisymmetric tensor in d=4 has 6 DOF, decomposing as:
#   spin-1: 3 DOF (magnetic-type)
#   spin-0: 3 DOF (electric-type)
#
# Projectors (4-index: maps B_{μν} to spin-s subspace):
#   P^(1)_{μν,ρσ} = (1/2)(θ_{μρ}θ_{νσ} - θ_{μσ}θ_{νρ})
#   P^(0)_{μν,ρσ} = (1/2)(θ_{μρ}ω_{νσ} - θ_{μσ}ω_{νρ}
#                        + ω_{μρ}θ_{νσ} - ω_{μσ}θ_{νρ})
#                  + ω_{μρ}ω_{νσ} - ω_{μσ}ω_{νρ}
#
# Completeness on antisymmetric subspace:
#   P^(1) + P^(0) = A_{μν,ρσ} = (1/2)(η_{μρ}η_{νσ} - η_{μσ}η_{νρ})
#
# Ground truth: PSALTer arXiv:2406.09500, Sec 3.2.
=#

"""
    antisym2_spin1_projector(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                              kwargs...) -> TensorExpr

Spin-1 projector for antisymmetric rank-2 field B_{[μν]}:

    P^(1)_{μν,ρσ} = (1/2)(θ_{μρ}θ_{νσ} - θ_{μσ}θ_{νρ})

Projects onto the 3-dimensional magnetic-type subspace (in d=4).

Ground truth: PSALTer arXiv:2406.09500, Sec 3.2.
"""
function antisym2_spin1_projector(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                                   kwargs...)
    θ_μρ = theta_projector(μ, ρ; kwargs...)
    θ_νσ = theta_projector(ν, σ; kwargs...)
    θ_μσ = theta_projector(μ, σ; kwargs...)
    θ_νρ = theta_projector(ν, ρ; kwargs...)

    tproduct(1 // 2, TensorExpr[θ_μρ * θ_νσ - θ_μσ * θ_νρ])
end

"""
    antisym2_spin0_projector(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                              kwargs...) -> TensorExpr

Spin-0 projector for antisymmetric rank-2 field B_{[μν]}:

    P^(0)_{μν,ρσ} = A_{μν,ρσ} - P^(1)_{μν,ρσ}

where A_{μν,ρσ} = (1/2)(η_{μρ}η_{νσ} - η_{μσ}η_{νρ}) is the
antisymmetric identity projector.

Projects onto the 3-dimensional electric-type subspace (in d=4).

Ground truth: PSALTer arXiv:2406.09500, Sec 3.2.
"""
function antisym2_spin0_projector(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                                   kwargs...)
    # A_{μν,ρσ} = (1/2)(η_{μρ}η_{νσ} - η_{μσ}η_{νρ})
    metric = get(kwargs, :metric, :g)
    η_μρ = Tensor(metric, [μ, ρ])
    η_νσ = Tensor(metric, [ν, σ])
    η_μσ = Tensor(metric, [μ, σ])
    η_νρ = Tensor(metric, [ν, ρ])

    A = tproduct(1 // 2, TensorExpr[η_μρ * η_νσ - η_μσ * η_νρ])
    P1 = antisym2_spin1_projector(μ, ν, ρ, σ; kwargs...)

    A - P1
end

"""
    antisym2_identity(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                      metric::Symbol=:g) -> TensorExpr

Antisymmetric identity projector:

    A_{μν,ρσ} = (1/2)(η_{μρ}η_{νσ} - η_{μσ}η_{νρ})

Satisfies A_{μν,ρσ} B^{ρσ} = B_{μν} for any antisymmetric B.
"""
function antisym2_identity(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                            metric::Symbol=:g)
    η_μρ = Tensor(metric, [μ, ρ])
    η_νσ = Tensor(metric, [ν, σ])
    η_μσ = Tensor(metric, [μ, σ])
    η_νρ = Tensor(metric, [ν, ρ])

    tproduct(1 // 2, TensorExpr[η_μρ * η_νσ - η_μσ * η_νρ])
end
