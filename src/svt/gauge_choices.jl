#= Cosmological perturbation gauge choices.
#
# Standard gauge fixings for cosmological perturbation theory,
# applicable to both FLRW and Bianchi backgrounds.
#
# Each gauge is implemented as a set of conditions that set certain
# perturbation variables to zero via set_vanishing! or rewrite rules.
#
# Ground truth: Bardeen, Phys. Rev. D 22, 1882 (1980);
#              Pitrou et al, JCAP 04 (2013) 004, Sec 2.3.
=#

"""
    GaugeChoice

A gauge choice for cosmological perturbation theory.

# Fields
- `name::Symbol`                  -- gauge name (:synchronous, :newtonian, etc.)
- `vanishing_fields::Vector{Symbol}` -- fields set to zero in this gauge
- `description::String`           -- human-readable description
"""
struct GaugeChoice
    name::Symbol
    vanishing_fields::Vector{Symbol}
    description::String
end

function Base.show(io::IO, gc::GaugeChoice)
    print(io, "GaugeChoice(:$(gc.name), sets ", gc.vanishing_fields, " = 0)")
end

# ────────────────────────────────────────────────────────────────────
# Standard gauge choices
# ────────────────────────────────────────────────────────────────────

"""
    synchronous_gauge(; fields::SVTFields=DEFAULT_SVT) -> GaugeChoice

Synchronous gauge: δg_{00} = δg_{0i} = 0.
In SVT variables: ϕ = 0, B = 0 (and vector S_i = 0).

Residual gauge freedom remains (spatial coordinate choice).

Ground truth: Bardeen (1980) Sec III; Pitrou et al Sec 2.3.
"""
function synchronous_gauge(; fields::SVTFields=DEFAULT_SVT)
    GaugeChoice(:synchronous,
        [fields.ϕ, fields.B, fields.S],
        "δg₀₀ = δg₀ᵢ = 0")
end

"""
    newtonian_gauge(; fields::SVTFields=DEFAULT_SVT) -> GaugeChoice

Newtonian (longitudinal, conformal) gauge: B = E = 0.
The scalar sector has only two potentials: Ψ (= ϕ) and Φ (= ψ).

Fully fixes scalar gauge freedom. No residual gauge in scalar sector.

Ground truth: Bardeen (1980) Sec IV; Mukhanov, Feldman & Brandenberger (1992).
"""
function newtonian_gauge(; fields::SVTFields=DEFAULT_SVT)
    GaugeChoice(:newtonian,
        [fields.B, fields.E],
        "B = E = 0 (longitudinal gauge)")
end

"""
    flat_slicing_gauge(; fields::SVTFields=DEFAULT_SVT) -> GaugeChoice

Flat slicing gauge: ψ = E = 0.
The spatial metric perturbation has no scalar trace or longitudinal part.

Useful for inflationary perturbation theory (curvature perturbation ζ = -ψ).

Ground truth: Pitrou et al Sec 2.3; Maldacena, JHEP 05 (2003) 013.
"""
function flat_slicing_gauge(; fields::SVTFields=DEFAULT_SVT)
    GaugeChoice(:flat_slicing,
        [fields.ψ, fields.E],
        "ψ = E = 0 (spatially flat slices)")
end

"""
    comoving_gauge(; fields::SVTFields=DEFAULT_SVT) -> GaugeChoice

Comoving gauge: B = 0, and matter velocity perturbation δv = 0.
In the absence of anisotropic stress: E = 0 also.

This is the gauge used for the comoving curvature perturbation ℛ.

Ground truth: Pitrou et al Sec 2.3; Lyth & Liddle (2009) Sec 14.
"""
function comoving_gauge(; fields::SVTFields=DEFAULT_SVT)
    GaugeChoice(:comoving,
        [fields.B],
        "B = 0 (comoving with matter)")
end

"""
    uniform_density_gauge(; fields::SVTFields=DEFAULT_SVT) -> GaugeChoice

Uniform density gauge: δρ = 0 (constant energy density slicing).

Sets the scalar field perturbation to zero on the spatial hypersurface.
Used in uniform-curvature computations.

Ground truth: Pitrou et al Sec 2.3.
"""
function uniform_density_gauge(; fields::SVTFields=DEFAULT_SVT)
    GaugeChoice(:uniform_density,
        Symbol[],  # δρ is a matter field, not an SVT field
        "δρ = 0 (uniform density slicing)")
end

# ────────────────────────────────────────────────────────────────────
# Gauge application
# ────────────────────────────────────────────────────────────────────

"""
    apply_gauge!(reg::TensorRegistry, gauge::GaugeChoice) -> Nothing

Apply a gauge choice by setting the specified fields to vanish.
Uses `set_vanishing!` for each field in the gauge's vanishing list.

Only sets vanishing for fields that are registered in the registry.
"""
function apply_gauge!(reg::TensorRegistry, gauge::GaugeChoice)
    for field in gauge.vanishing_fields
        if has_tensor(reg, field)
            set_vanishing!(reg, field)
        end
    end
    nothing
end

"""
    gauge_choice(name::Symbol; fields::SVTFields=DEFAULT_SVT) -> GaugeChoice

Look up a standard gauge choice by name.

Supported gauges:
- `:synchronous`    -- δg₀₀ = δg₀ᵢ = 0
- `:newtonian`      -- B = E = 0 (longitudinal)
- `:flat_slicing`   -- ψ = E = 0
- `:comoving`       -- B = 0
- `:uniform_density` -- δρ = 0
"""
function gauge_choice(name::Symbol; fields::SVTFields=DEFAULT_SVT)
    name == :synchronous    && return synchronous_gauge(; fields=fields)
    name == :newtonian      && return newtonian_gauge(; fields=fields)
    name == :flat_slicing   && return flat_slicing_gauge(; fields=fields)
    name == :comoving       && return comoving_gauge(; fields=fields)
    name == :uniform_density && return uniform_density_gauge(; fields=fields)
    error("Unknown gauge choice: :$name. Supported: :synchronous, :newtonian, :flat_slicing, :comoving, :uniform_density")
end
