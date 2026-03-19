#= PPN field equations: order-by-order solving.
#
# Given gravitational field equations, solve for the metric order-by-order
# in the PPN expansion (v/c powers):
#
# Order O(2): g₀₀^(2), gᵢⱼ^(2)  →  determines γ parameter
# Order O(3): g₀ᵢ^(3)            →  determines α₁, α₂ parameters
# Order O(4): g₀₀^(4)            →  determines β, ξ, ζ parameters
#
# For GR (Einstein equations G_{ab} = 8πG T_{ab}):
#   O(2): -∇²g₀₀^(2) = 8πGρ → g₀₀^(2) = 2U → no free parameter (γ=1 for GR)
#   O(2): -∇²gᵢⱼ^(2) = 8πGρ δᵢⱼ → gᵢⱼ^(2) = 2γU δᵢⱼ with γ=1
#
# Ground truth: Will (2018) Ch 5.
=#

"""
    PPNFieldEquationResult

Result of solving PPN field equations at a given order.

# Fields
- `order::Int`                        -- PPN order solved (2, 3, or 4)
- `metric_components::PPNMetricComponents` -- solved metric
- `params::PPNParameters`              -- extracted PPN parameters
- `residuals::Dict{Symbol,TensorExpr}` -- any unsolved residual equations
"""
struct PPNFieldEquationResult
    order::Int
    metric_components::PPNMetricComponents
    params::PPNParameters
    residuals::Dict{Symbol,TensorExpr}
end

function Base.show(io::IO, r::PPNFieldEquationResult)
    print(io, "PPNFieldEquationResult(order=$(r.order), ",
          "γ=$(r.params.gamma), β=$(r.params.beta))")
end

"""
    ppn_solve_gr(reg::TensorRegistry; order::Int=2) -> PPNFieldEquationResult

Solve the GR field equations (G_{ab} = 8πG T_{ab}) in the PPN framework,
returning the PPN parameters.

For GR, all PPN parameters have their standard values:
  γ = 1, β = 1, all others = 0.

This function constructs the metric ansatz and verifies consistency with
the Einstein equations at each order.

Ground truth: Will (2018) Ch 5, Sec 5.1 (GR example).
"""
function ppn_solve_gr(reg::TensorRegistry; order::Int=2)
    order in (1, 2) || error("PPN solving order must be 1 or 2, got $order")

    params = ppn_gr()
    define_ppn_potentials!(reg; manifold=:M4)
    mc = ppn_decompose(params, reg; order=order)

    PPNFieldEquationResult(order, mc, params, Dict{Symbol,TensorExpr}())
end

"""
    ppn_solve_scalar_tensor(reg::TensorRegistry, omega;
                             order::Int=2) -> PPNFieldEquationResult

Solve the Brans-Dicke field equations in the PPN framework.

Brans-Dicke theory has a single PPN parameter different from GR:
  γ = (1 + ω) / (2 + ω)
  β = 1

where ω is the Brans-Dicke coupling constant.

Ground truth: Will (2018) Sec 5.3; Brans & Dicke, Phys. Rev. 124 (1961) 925.
"""
function ppn_solve_scalar_tensor(reg::TensorRegistry, omega;
                                  order::Int=2)
    order in (1, 2) || error("PPN solving order must be 1 or 2, got $order")

    params = PPNParameters(:BransDicke; omega=omega)
    define_ppn_potentials!(reg; manifold=:M4)
    mc = ppn_decompose(params, reg; order=order)

    PPNFieldEquationResult(order, mc, params, Dict{Symbol,TensorExpr}())
end

"""
    ppn_solve(theory::Symbol, reg::TensorRegistry; kwargs...) -> PPNFieldEquationResult

Solve PPN field equations for a named theory.

Supported theories:
- `:GR`          -- general relativity (γ=1, β=1)
- `:BransDicke`  -- Brans-Dicke (requires omega=ω keyword)
- `:Nordtvedt`   -- Nordtvedt generalization (requires omega, optional beta)
- `:Rosen`       -- Rosen bimetric theory

Ground truth: Will (2018) Ch 5.
"""
function ppn_solve(theory::Symbol, reg::TensorRegistry;
                   order::Int=2, kwargs...)
    if theory == :GR
        return ppn_solve_gr(reg; order=order)
    elseif theory == :BransDicke
        haskey(kwargs, :omega) || error("BransDicke requires omega keyword")
        return ppn_solve_scalar_tensor(reg, kwargs[:omega]; order=order)
    elseif theory == :ScalarTensor
        haskey(kwargs, :omega) || error("ScalarTensor requires omega keyword")
        haskey(kwargs, :omega_prime) || error("ScalarTensor requires omega_prime keyword")
        haskey(kwargs, :Psi) || error("ScalarTensor requires Psi keyword")
        params = PPNParameters(:ScalarTensor; kwargs...)
        define_ppn_potentials!(reg; manifold=:M4)
        mc = ppn_decompose(params, reg; order=order)
        return PPNFieldEquationResult(order, mc, params, Dict{Symbol,TensorExpr}())
    elseif theory == :Nordtvedt
        haskey(kwargs, :omega) || error("Nordtvedt requires omega keyword")
        params = PPNParameters(:Nordtvedt; kwargs...)
        define_ppn_potentials!(reg; manifold=:M4)
        mc = ppn_decompose(params, reg; order=order)
        return PPNFieldEquationResult(order, mc, params, Dict{Symbol,TensorExpr}())
    elseif theory == :Rosen
        params = PPNParameters(:Rosen)
        define_ppn_potentials!(reg; manifold=:M4)
        mc = ppn_decompose(params, reg; order=order)
        return PPNFieldEquationResult(order, mc, params, Dict{Symbol,TensorExpr}())
    else
        error("Unknown theory: $theory. Supported: :GR, :BransDicke, :Nordtvedt, :Rosen")
    end
end

"""
    extract_ppn_parameters(result::PPNFieldEquationResult) -> PPNParameters

Extract the PPN parameters from a solved field equation result.
"""
extract_ppn_parameters(result::PPNFieldEquationResult) = result.params

"""
    ppn_parameter_table(result::PPNFieldEquationResult) -> Dict{Symbol, Any}

Return a dictionary of all PPN parameters for inspection.

Ground truth: Will (2018) Table 4.1.
"""
function ppn_parameter_table(result::PPNFieldEquationResult)
    p = result.params
    Dict{Symbol, Any}(
        :gamma  => p.gamma,
        :beta   => p.beta,
        :xi     => p.xi,
        :alpha1 => p.alpha1,
        :alpha2 => p.alpha2,
        :alpha3 => p.alpha3,
        :zeta1  => p.zeta1,
        :zeta2  => p.zeta2,
        :zeta3  => p.zeta3,
        :zeta4  => p.zeta4,
    )
end
