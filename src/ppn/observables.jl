#= PPN observable expressions.
#
# Closed-form expressions for solar system tests as functions of PPN
# parameters. These connect abstract PPN parameters to measurable quantities.
#
# Ground truth: Will (2018) Ch 7-8.
=#

"""
    ppn_perihelion(params::PPNParameters; M=:M, a=:a, e=:e) -> Any

Perihelion precession per orbit in the PPN framework:

    δω = (2 + 2γ - β) × 6π M / [a(1 - e²)]

For GR (γ=1, β=1): δω = 6πM/[a(1-e²)] = 42.98"/century for Mercury.

Ground truth: Will (2018) Eq 7.30.
"""
function ppn_perihelion(params::PPNParameters; M=:M, a=:a, e=:e)
    factor = 2 + 2 * params.gamma - params.beta
    (factor, :*, 6, :*, :pi, :*, M, :/, (a, :*, (1, :-, e, :^, 2)))
end

"""
    ppn_perihelion_factor(params::PPNParameters) -> Any

The PPN combination that appears in perihelion precession:

    (2 + 2γ - β) / 3

For GR: (2+2-1)/3 = 1. Observational constraint: ≈ 1 ± 10⁻³.

Ground truth: Will (2018) Eq 7.30.
"""
function ppn_perihelion_factor(params::PPNParameters)
    (2 + 2 * params.gamma - params.beta) / 3
end

"""
    ppn_deflection(params::PPNParameters; M=:M, d=:d) -> Any

Light deflection angle in the PPN framework:

    δθ = (1 + γ)/2 × 4M/d

For GR (γ=1): δθ = 4M/d = 1.75" for the Sun.

Ground truth: Will (2018) Eq 7.13.
"""
function ppn_deflection(params::PPNParameters; M=:M, d=:d)
    factor = (1 + params.gamma) / 2
    (factor, :*, 4, :*, M, :/, d)
end

"""
    ppn_deflection_factor(params::PPNParameters) -> Any

The PPN combination that appears in light deflection:

    (1 + γ) / 2

For GR: (1+1)/2 = 1. Best current bound: Cassini (2003), |γ-1| < 2.3×10⁻⁵.

Ground truth: Will (2018) Eq 7.13.
"""
ppn_deflection_factor(params::PPNParameters) = (1 + params.gamma) / 2

"""
    ppn_shapiro_delay(params::PPNParameters; M=:M, r1=:r1, r2=:r2, d=:d) -> Any

Shapiro time delay in the PPN framework:

    Δt = (1 + γ) × 2M × ln(4r₁r₂/d²)

For GR (γ=1): Δt = 4M ln(4r₁r₂/d²).

Ground truth: Will (2018) Eq 7.20; Shapiro, PRL 13, 789 (1964).
"""
function ppn_shapiro_delay(params::PPNParameters; M=:M, r1=:r1, r2=:r2, d=:d)
    factor = 1 + params.gamma
    (factor, :*, 2, :*, M, :*, :log, (4, :*, r1, :*, r2, :/, d, :^, 2))
end

"""
    ppn_shapiro_factor(params::PPNParameters) -> Any

The PPN combination in Shapiro delay: (1 + γ)/2.

Same as deflection factor. For GR: 1.

Ground truth: Will (2018) Eq 7.20.
"""
ppn_shapiro_factor(params::PPNParameters) = (1 + params.gamma) / 2

"""
    ppn_nordtvedt_eta(params::PPNParameters) -> Any

The Nordtvedt parameter η_N:

    η_N = 4β - γ - 3 - (10/3)ξ - α₁ + (2/3)α₂ - (2/3)ζ₁ - (1/3)ζ₂

For GR: η_N = 4-1-3 = 0. A nonzero η_N implies a violation of the
Strong Equivalence Principle (SEP) via the Nordtvedt effect: bodies
with different gravitational self-energy fall at different rates.

Ground truth: Will (2018) Eq 8.20; Nordtvedt, Phys. Rev. 169, 1017 (1968).
"""
function ppn_nordtvedt_eta(params::PPNParameters)
    4 * params.beta - params.gamma - 3 -
        (10 // 3) * params.xi -
        params.alpha1 +
        (2 // 3) * params.alpha2 -
        (2 // 3) * params.zeta1 -
        (1 // 3) * params.zeta2
end

"""
    ppn_geodetic_precession(params::PPNParameters; M=:M, r=:r) -> Any

Geodetic (de Sitter) precession rate:

    Ω_gp = (γ + 1/2) × M/r  per orbit

For GR (γ=1): Ω_gp = (3/2) M/r.

Measured by Gravity Probe B (2011): confirmed to 0.28%.

Ground truth: Will (2018) Eq 9.7.
"""
function ppn_geodetic_precession(params::PPNParameters; M=:M, r=:r)
    factor = params.gamma + 1 // 2
    (factor, :*, M, :/, r)
end

"""
    ppn_geodetic_factor(params::PPNParameters) -> Any

The PPN combination in geodetic precession: (γ + 1/2).

For GR: 3/2.

Ground truth: Will (2018) Eq 9.7.
"""
ppn_geodetic_factor(params::PPNParameters) = params.gamma + 1 // 2

"""
    ppn_is_fully_conservative(params::PPNParameters) -> Bool

Check if the theory is fully conservative (no preferred-frame or
conservation-law-violating effects):

    α₁ = α₂ = α₃ = ζ₁ = ζ₂ = ζ₃ = ζ₄ = 0

Ground truth: Will (2018) Sec 4.2.
"""
ppn_is_fully_conservative(params::PPNParameters) = is_fully_conservative(params)

"""
    ppn_observational_bounds() -> Dict{Symbol, Tuple{Any, String}}

Return current observational bounds on PPN parameters.

Each entry: parameter => (bound_value, experiment).

Ground truth: Will (2018) Table 4.1 (updated to ~2018 values).
"""
function ppn_observational_bounds()
    Dict{Symbol, Tuple{Any, String}}(
        :gamma  => (2.3e-5, "Cassini 2003"),
        :beta   => (3e-3,   "Lunar laser ranging"),
        :xi     => (4e-9,   "Millisecond pulsars"),
        :alpha1 => (4e-5,   "Lunar laser ranging + binary pulsars"),
        :alpha2 => (2e-9,   "Solar alignment with ecliptic"),
        :alpha3 => (4e-20,  "Pulsar spin-down statistics"),
        :zeta1  => (2e-2,   "Combined PPN"),
        :zeta2  => (4e-5,   "Binary acceleration"),
        :zeta3  => (1e-8,   "Newton's 3rd law tests"),
        :zeta4  => (0.006,  "Kreuzer experiment (indirect)"),
    )
end
