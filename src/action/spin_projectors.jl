#= Barnes-Rivers spin-projection operators for symmetric rank-2 fields.

In momentum space with 4-momentum k^μ, define:
  θ_{μν} = η_{μν} - k_μ k_ν / k²   (transverse projector)
  ω_{μν} = k_μ k_ν / k²             (longitudinal projector)

The 6 Barnes-Rivers projectors decompose symmetric rank-2 fields h_{μν}
into spin sectors:
  P^(2)     spin-2 (TT part)
  P^(1)     spin-1 (vector part)
  P^(0-s)   spin-0 scalar (transverse trace)
  P^(0-w)   spin-0 w (longitudinal)
  T^(sw)    transfer operator (0-s → 0-w)
  T^(ws)    transfer operator (0-w → 0-s)

Reference: Barnes & Rivers (1963); PSALTer (Barker+ 2024)

## Usage on maximally symmetric spaces (dS/AdS)

On a maximally symmetric space (MSS) with R_{μν} = Λg_{μν}, the flat
Barnes-Rivers projectors are algebraically correct (they still form a
complete, orthogonal, idempotent basis for symmetric rank-2 tensors).
However, spin_project(K, :spin2) gives the FULL kinetic operator trace
including the Lichnerowicz mass term, not just the kinetic form factor.

For the Einstein-Hilbert action on MSS:
  Tr(K_EH · P²_flat) = (5/2)(k² − 2Λ)     [full operator including mass]
  bc_to_form_factors(bc_EH(κ,Λ)) = (5/2)κk²  [kinetic form factor only]

The discrepancy of −5Λ is the spin-2 Lichnerowicz mass on MSS, which is
physical but should be separated from the kinetic form factor for spectrum
analysis (propagator poles, ghost conditions, unitarity bounds).

Use `spin_project_mss` (in kernel_extraction.jl) to automatically extract
the kinetic form factor by subtracting the k²-independent mass term.
This matches the Bueno-Cano convention (arXiv:1607.06463).
=#

"""
    theta_projector(μ, ν; metric=:η, k_name=:k, k_sq=:k²) -> TensorExpr

Transverse projector: θ_{μν} = η_{μν} - k_μ k_ν / k²
"""
function theta_projector(μ::TIndex, ν::TIndex;
                          metric::Symbol=:g, k_name::Symbol=:k, k_sq=:k²)
    η = Tensor(metric, [μ, ν])
    k_μ = Tensor(k_name, [μ])
    k_ν = Tensor(k_name, [ν])
    η - TProduct(1 // 1, TensorExpr[TScalar(_sym_div(1, k_sq)), k_μ, k_ν])
end

"""
    omega_projector(μ, ν; k_name=:k, k_sq=:k²) -> TensorExpr

Longitudinal projector: ω_{μν} = k_μ k_ν / k²
"""
function omega_projector(μ::TIndex, ν::TIndex;
                          k_name::Symbol=:k, k_sq=:k²)
    k_μ = Tensor(k_name, [μ])
    k_ν = Tensor(k_name, [ν])
    TProduct(1 // 1, TensorExpr[TScalar(_sym_div(1, k_sq)), k_μ, k_ν])
end

"""
    spin2_projector(μ, ν, ρ, σ; dim=4, kwargs...) -> TensorExpr

Spin-2 Barnes-Rivers projector:
P^(2)_{μν,ρσ} = (1/2)(θ_{μρ}θ_{νσ} + θ_{μσ}θ_{νρ}) - (1/(d-1))θ_{μν}θ_{ρσ}
"""
function spin2_projector(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                          dim::Int=4, kwargs...)
    θ_μρ = theta_projector(μ, ρ; kwargs...)
    θ_νσ = theta_projector(ν, σ; kwargs...)
    θ_μσ = theta_projector(μ, σ; kwargs...)
    θ_νρ = theta_projector(ν, ρ; kwargs...)
    θ_μν = theta_projector(μ, ν; kwargs...)
    θ_ρσ = theta_projector(ρ, σ; kwargs...)

    (1 // 2) * (θ_μρ * θ_νσ + θ_μσ * θ_νρ) -
    (1 // (dim - 1)) * θ_μν * θ_ρσ
end

"""
    spin1_projector(μ, ν, ρ, σ; kwargs...) -> TensorExpr

Spin-1 Barnes-Rivers projector:
P^(1)_{μν,ρσ} = (1/2)(θ_{μρ}ω_{νσ} + θ_{μσ}ω_{νρ} + θ_{νρ}ω_{μσ} + θ_{νσ}ω_{μρ})
"""
function spin1_projector(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex; kwargs...)
    θ_μρ = theta_projector(μ, ρ; kwargs...)
    θ_μσ = theta_projector(μ, σ; kwargs...)
    θ_νρ = theta_projector(ν, ρ; kwargs...)
    θ_νσ = theta_projector(ν, σ; kwargs...)
    ω_μσ = omega_projector(μ, σ; _omega_kwargs(kwargs)...)
    ω_νσ = omega_projector(ν, σ; _omega_kwargs(kwargs)...)
    ω_μρ = omega_projector(μ, ρ; _omega_kwargs(kwargs)...)
    ω_νρ = omega_projector(ν, ρ; _omega_kwargs(kwargs)...)

    (1 // 2) * (θ_μρ * ω_νσ + θ_μσ * ω_νρ + θ_νρ * ω_μσ + θ_νσ * ω_μρ)
end

"""
    spin0s_projector(μ, ν, ρ, σ; dim=4, kwargs...) -> TensorExpr

Spin-0-s (scalar) Barnes-Rivers projector:
P^(0-s)_{μν,ρσ} = (1/(d-1)) θ_{μν} θ_{ρσ}
"""
function spin0s_projector(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                           dim::Int=4, kwargs...)
    θ_μν = theta_projector(μ, ν; kwargs...)
    θ_ρσ = theta_projector(ρ, σ; kwargs...)
    (1 // (dim - 1)) * θ_μν * θ_ρσ
end

"""
    spin0w_projector(μ, ν, ρ, σ; kwargs...) -> TensorExpr

Spin-0-w (longitudinal) Barnes-Rivers projector:
P^(0-w)_{μν,ρσ} = ω_{μν} ω_{ρσ}
"""
function spin0w_projector(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex; kwargs...)
    ω_μν = omega_projector(μ, ν; _omega_kwargs(kwargs)...)
    ω_ρσ = omega_projector(ρ, σ; _omega_kwargs(kwargs)...)
    ω_μν * ω_ρσ
end

"""
    transfer_sw(μ, ν, ρ, σ; dim=4, kwargs...) -> TensorExpr

Transfer operator T^(sw): maps spin-0-w to spin-0-s sector.
T^(sw)_{μν,ρσ} = (1/(d-1)) θ_{μν} ω_{ρσ}

Note: T^(sw) · T^(ws) = (1/(d-1)) P^(0-s), not idempotent.
Use with normalization √(d-1) for standard Barnes-Rivers convention.
"""
function transfer_sw(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                      dim::Int=4, kwargs...)
    θ_μν = theta_projector(μ, ν; kwargs...)
    ω_ρσ = omega_projector(ρ, σ; _omega_kwargs(kwargs)...)
    (1 // (dim - 1)) * θ_μν * ω_ρσ
end

"""
    transfer_ws(μ, ν, ρ, σ; dim=4, kwargs...) -> TensorExpr

Transfer operator T^(ws): maps spin-0-s to spin-0-w sector.
T^(ws)_{μν,ρσ} = (1/(d-1)) ω_{μν} θ_{ρσ}
"""
function transfer_ws(μ::TIndex, ν::TIndex, ρ::TIndex, σ::TIndex;
                      dim::Int=4, kwargs...)
    ω_μν = omega_projector(μ, ν; _omega_kwargs(kwargs)...)
    θ_ρσ = theta_projector(ρ, σ; kwargs...)
    (1 // (dim - 1)) * ω_μν * θ_ρσ
end

# Helper: extract omega-compatible kwargs (drop metric)
function _omega_kwargs(kwargs)
    pairs = Pair{Symbol,Any}[]
    for (k, v) in kwargs
        k == :metric && continue
        k == :dim && continue
        push!(pairs, k => v)
    end
    pairs
end
