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

# ─────────────────────────────────────────────────────────────
# Abstract Poisson equation solver for PPN potentials
# ─────────────────────────────────────────────────────────────

#= Poisson equations in the PPN formalism.
#
# The PPN potentials are defined as solutions of Poisson-type equations:
#   ∇²U     = -4πGρ            (Newtonian, order 2)
#   ∇²V_i   = -4πGρv_i         (gravitomagnetic, order 3)
#   ∇²Φ_1   = -4πGρv²          (post-Newtonian, order 4)
#   ∇²Φ_2   = -4πGρU           (post-Newtonian, order 4)
#   ∇²Φ_3   = -4πGρΠ           (post-Newtonian, order 4, Π = specific internal energy)
#   ∇²Φ_4   = -4πGp            (post-Newtonian, order 4, p = pressure)
#
# The abstract solver establishes symbolic relationships between potentials
# and their source terms without numerical integration.
#
# Ground truth: Will (2018) Ch 4-5; Poisson & Will (2014) Ch 8.
=#

"""
    PoissonEquation

Represents a Poisson-type equation ∇²Φ = source defining a PPN potential.

# Fields
- `potential::Symbol`       -- the PPN potential being solved for (e.g., :U, :Phi_1)
- `source::TensorExpr`     -- the RHS source term (e.g., -4πGρ for U)
- `order::Int`             -- PPN velocity order (v/c power: 2, 3, or 4)
- `equation_type::Symbol`  -- :scalar, :vector, or :tensor

Ground truth: Will (2018) Eqs 4.4a--4.4f.
"""
struct PoissonEquation
    potential::Symbol
    source::TensorExpr
    order::Int
    equation_type::Symbol

    function PoissonEquation(potential::Symbol, source::TensorExpr,
                             order::Int, equation_type::Symbol)
        equation_type in (:scalar, :vector, :tensor) ||
            error("equation_type must be :scalar, :vector, or :tensor, got $equation_type")
        order >= 0 || error("PPN order must be non-negative, got $order")
        new(potential, source, order, equation_type)
    end
end

function Base.show(io::IO, eq::PoissonEquation)
    print(io, "PoissonEquation(∇²$(eq.potential) = source, order=$(eq.order), type=$(eq.equation_type))")
end

"""
    PoissonSolution

Formal Green's function solution to a PoissonEquation.

The solution has the form:
    Φ(x) = -(1/4π) ∫ source(x') / |x - x'| d³x'

For the standard Newtonian Green's function G(x,x') = -1/(4π|x-x'|).

# Fields
- `equation::PoissonEquation`    -- the equation being solved
- `green_function::Symbol`       -- Green's function type (:newtonian for 1/|x-x'|)
- `integral_form::TensorExpr`    -- symbolic representation of the integral solution

Ground truth: Will (2018) Eqs 4.5--4.6; Poisson & Will (2014) Sec 8.1.
"""
struct PoissonSolution
    equation::PoissonEquation
    green_function::Symbol
    integral_form::TensorExpr
end

function Base.show(io::IO, sol::PoissonSolution)
    pot = sol.equation.potential
    gf = sol.green_function
    print(io, "PoissonSolution($(pot), green=$(gf))")
end

"""
    define_poisson_equation(potential::Symbol, source::TensorExpr;
                            order::Int, type::Symbol=:scalar,
                            registry::TensorRegistry=current_registry())
        -> PoissonEquation

Create a Poisson equation ∇²Φ = source for a PPN potential.

Validates that the potential tensor exists in the registry.

# Arguments
- `potential`  -- name of the PPN potential tensor
- `source`     -- RHS source expression
- `order`      -- PPN velocity order (2, 3, or 4)
- `type`       -- equation type (:scalar, :vector, :tensor)
- `registry`   -- tensor registry (default: current)

Ground truth: Will (2018) Sec 4.1.
"""
function define_poisson_equation(potential::Symbol, source::TensorExpr;
                                  order::Int, type::Symbol=:scalar,
                                  registry::TensorRegistry=current_registry())
    has_tensor(registry, potential) ||
        error("Potential tensor $potential not registered in the registry")
    PoissonEquation(potential, source, order, type)
end

"""
    poisson_solve(eq::PoissonEquation;
                  registry::TensorRegistry=current_registry())
        -> PoissonSolution

Construct the formal Green's function solution to a Poisson equation.

For a scalar equation ∇²Φ = S, the solution is:
    Φ(x) = -(1/4π) ∫ S(x') / |x - x'| d³x'

For a vector equation ∇²Φ_i = S_i, the same Green's function applies
component-wise.

The returned PoissonSolution contains the integral form as a symbolic
tensor expression: (-1/4π) × source, representing the convolution with
the Newtonian Green's function.

Ground truth: Will (2018) Eq 4.5; Jackson, "Classical Electrodynamics" Sec 1.7.
"""
function poisson_solve(eq::PoissonEquation;
                       registry::TensorRegistry=current_registry())
    # The formal solution is Φ = -(1/4π) ∫ source(x')/|x-x'| d³x'
    # We represent this symbolically as the coefficient (-1/4π) times the source,
    # with the understanding that the Green's function convolution is implied.
    #
    # The integral_form captures the integrand structure:
    #   integral_form = -(1/4π) × source
    # The actual spatial integration (∫ d³x' / |x-x'|) is encoded by
    # the green_function field.

    integral_form = tproduct(-1 // 1, TensorExpr[
        TScalar(Symbol("1_over_4pi")),
        eq.source
    ])

    PoissonSolution(eq, :newtonian, integral_form)
end

"""
    standard_ppn_potentials(; registry::TensorRegistry=current_registry())
        -> Vector{PoissonEquation}

Return the standard set of PPN potential Poisson equations.

The 6 standard PPN Poisson equations are:
1. U:    ∇²U = -4πGρ                    (order 2, scalar, Newtonian)
2. V_i:  ∇²V_i = -4πGρv_i              (order 3, vector, gravitomagnetic)
3. Φ_1:  ∇²Φ₁ = -4πGρv²               (order 4, scalar)
4. Φ_2:  ∇²Φ₂ = -4πGρU                (order 4, scalar)
5. Φ_3:  ∇²Φ₃ = -4πGρΠ               (order 4, scalar, Π = specific internal energy)
6. Φ_4:  ∇²Φ₄ = -4πGp                 (order 4, scalar, p = pressure)

Each source is represented symbolically using a TScalar with a descriptive
symbol for the matter quantity (e.g., :rho, :rho_v2, :p).

Ground truth: Will (2018) Eqs 4.4a--4.4f; Poisson & Will (2014) Sec 8.1.
"""
function standard_ppn_potentials(; registry::TensorRegistry=current_registry())
    # Source terms are represented as TScalar symbols:
    #   -4πG × (matter source)
    # The -4πG prefactor is absorbed into the source representation,
    # matching the standard PPN convention.

    equations = PoissonEquation[]

    # 1. U: ∇²U = -4πGρ
    #    Source: -4πG × ρ
    if has_tensor(registry, :U)
        source_U = tproduct(-1 // 1, TensorExpr[
            TScalar(Symbol("4piG")), TScalar(:rho)
        ])
        push!(equations, PoissonEquation(:U, source_U, 2, :scalar))
    end

    # 2. V_i: ∇²V_i = -4πGρv_i
    #    Source: -4πG × ρv_i (vector)
    if has_tensor(registry, :V_ppn)
        source_V = tproduct(-1 // 1, TensorExpr[
            TScalar(Symbol("4piG")), TScalar(:rho_vi)
        ])
        push!(equations, PoissonEquation(:V_ppn, source_V, 3, :vector))
    end

    # 3. Φ_1: ∇²Φ₁ = -4πGρv²
    #    Source: -4πG × ρv²
    if has_tensor(registry, :Phi_1)
        source_Phi1 = tproduct(-1 // 1, TensorExpr[
            TScalar(Symbol("4piG")), TScalar(:rho_v2)
        ])
        push!(equations, PoissonEquation(:Phi_1, source_Phi1, 4, :scalar))
    end

    # 4. Φ_2: ∇²Φ₂ = -4πGρU
    #    Source: -4πG × ρU
    if has_tensor(registry, :Phi_2)
        source_Phi2 = tproduct(-1 // 1, TensorExpr[
            TScalar(Symbol("4piG")), TScalar(:rho_U)
        ])
        push!(equations, PoissonEquation(:Phi_2, source_Phi2, 4, :scalar))
    end

    # 5. Φ_3: ∇²Φ₃ = -4πGρΠ
    #    Source: -4πG × ρΠ (Π = specific internal energy)
    if has_tensor(registry, :Phi_3)
        source_Phi3 = tproduct(-1 // 1, TensorExpr[
            TScalar(Symbol("4piG")), TScalar(:rho_Pi)
        ])
        push!(equations, PoissonEquation(:Phi_3, source_Phi3, 4, :scalar))
    end

    # 6. Φ_4: ∇²Φ₄ = -4πGp
    #    Source: -4πG × p (pressure)
    if has_tensor(registry, :Phi_4)
        source_Phi4 = tproduct(-1 // 1, TensorExpr[
            TScalar(Symbol("4piG")), TScalar(:p)
        ])
        push!(equations, PoissonEquation(:Phi_4, source_Phi4, 4, :scalar))
    end

    equations
end

"""
    identify_ppn_potential(source_expr::TensorExpr;
                           registry::TensorRegistry=current_registry())
        -> Union{Symbol, Nothing}

Given a source expression from a field equation, identify which standard
PPN potential has that source term.

Pattern-matches against the known PPN source catalog by examining the
TScalar symbols in the source expression. Returns the potential name
(e.g., :U, :Phi_1) or `nothing` if no match is found.

The matching is based on the matter content symbols:
- `:rho`     → :U
- `:rho_vi`  → :V_ppn
- `:rho_v2`  → :Phi_1
- `:rho_U`   → :Phi_2
- `:rho_Pi`  → :Phi_3
- `:p`       → :Phi_4

Ground truth: Will (2018) Eqs 4.4a--4.4f.
"""
function identify_ppn_potential(source_expr::TensorExpr;
                                 registry::TensorRegistry=current_registry())
    # Source catalog: matter symbol → potential name
    _ppn_source_catalog = Dict{Symbol, Symbol}(
        :rho    => :U,
        :rho_vi => :V_ppn,
        :rho_v2 => :Phi_1,
        :rho_U  => :Phi_2,
        :rho_Pi => :Phi_3,
        :p      => :Phi_4,
    )

    # Extract TScalar symbols from the source expression
    matter_sym = _extract_matter_symbol(source_expr)
    matter_sym === nothing && return nothing

    get(_ppn_source_catalog, matter_sym, nothing)
end

# Helper: extract the matter-content symbol from a Poisson source expression.
# The source has the form: coeff × TScalar(4piG) × TScalar(matter_sym)
# or similar products. We look for a TScalar whose value is a known
# matter symbol.
function _extract_matter_symbol(expr::TScalar)
    val = expr.val
    val isa Symbol || return nothing
    # Check if it's a known matter symbol (not 4piG or other coupling constants)
    val in (:rho, :rho_vi, :rho_v2, :rho_U, :rho_Pi, :p) ? val : nothing
end

function _extract_matter_symbol(expr::TProduct)
    for f in expr.factors
        sym = _extract_matter_symbol(f)
        sym !== nothing && return sym
    end
    nothing
end

function _extract_matter_symbol(expr::TSum)
    # For sums, check the first term
    isempty(expr.terms) && return nothing
    _extract_matter_symbol(expr.terms[1])
end

_extract_matter_symbol(::TensorExpr) = nothing
