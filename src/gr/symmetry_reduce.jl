#= Symmetry-reduced metric ansatz generation.
#
# Given a symmetry ansatz (SphericalSymmetry, StaticSymmetry, etc.),
# generate the most general metric compatible with those symmetries.
#
# The reduced metric has fewer independent components than the general
# d×d metric. The remaining components are functions of the "essential"
# coordinates (those not killed by the symmetry).
#
# Examples:
# - Static + Spherical (Schwarzschild): ds² = -A(r)dt² + B(r)dr² + r²dΩ²
#   → 2 free functions of r (from 10 components in 4D)
# - Homogeneous + Isotropic (FLRW): ds² = -dt² + a(t)²(dr² + ...)
#   → 1 free function of t (the scale factor)
#
# References:
#   Wald, *General Relativity* (1984), Ch 5, 7.
#   Carroll, *Spacetime and Geometry* (2004), Ch 8.
=#

"""
    MetricAnsatzResult

Result of generating a symmetry-reduced metric.

# Fields
- `components::Dict{Tuple{Int,Int}, Any}` -- non-zero metric components g_{ij}
  (symmetric: only (i,j) with i ≤ j stored)
- `coords::Vector{Symbol}` -- coordinate names
- `free_functions::Vector{Symbol}` -- names of undetermined functions
- `essential_coords::Vector{Symbol}` -- coordinates the free functions depend on
- `symmetry::SymmetryAnsatz` -- the symmetry that was imposed
- `dim::Int` -- dimension
"""
struct MetricAnsatzResult
    components::Dict{Tuple{Int,Int}, Any}
    coords::Vector{Symbol}
    free_functions::Vector{Symbol}
    essential_coords::Vector{Symbol}
    symmetry::SymmetryAnsatz
    dim::Int
end

function Base.show(io::IO, r::MetricAnsatzResult)
    ncomp = length(r.components)
    nfree = length(r.free_functions)
    print(io, "MetricAnsatz(", r.symmetry, "): ", ncomp,
          " components, ", nfree, " free functions of ",
          join(r.essential_coords, ","))
end

# ── Static + Spherically Symmetric ────────────────────────────────────

"""
    symmetry_reduce(::SphericalSymmetry; static=true,
                     coords=[:t, :r, :theta, :phi]) -> MetricAnsatzResult

Generate the most general static, spherically symmetric metric:

    ds² = -A(r) dt² + B(r) dr² + r² dΩ²

where dΩ² = dθ² + sin²θ dφ² is the unit sphere metric.

The free functions A(r) and B(r) are determined by the Einstein equations.
For vacuum (Schwarzschild): A(r) = B(r)⁻¹ = 1 - 2M/r.

# Returns
A `MetricAnsatzResult` with 4 non-zero components and 2 free functions.
"""
function symmetry_reduce(sym::SphericalSymmetry;
                          static::Bool=true,
                          coords::Vector{Symbol}=[:t, :r, :theta, :phi])
    length(coords) == 4 ||
        error("symmetry_reduce(SphericalSymmetry): need 4 coordinates")

    t, r, theta, phi = coords

    # g_{tt} = -A(r), g_{rr} = B(r), g_{θθ} = r², g_{φφ} = r²sin²θ
    components = Dict{Tuple{Int,Int}, Any}(
        (1, 1) => (:neg, :A_metric),    # -A(r)
        (2, 2) => :B_metric,             # B(r)
        (3, 3) => (:r_squared,),         # r²
        (4, 4) => (:r2_sin2theta,),      # r²sin²θ
    )

    MetricAnsatzResult(
        components, coords,
        [:A_metric, :B_metric],  # free functions
        [r],                      # essential coordinate
        sym, 4
    )
end

# ── Static Symmetry alone ────────────────────────────────────────────

"""
    symmetry_reduce(::StaticSymmetry;
                     coords=[:t, :x, :y, :z]) -> MetricAnsatzResult

Generate the most general static metric:

    ds² = -N²(x,y,z) dt² + γ_{ij}(x,y,z) dx^i dx^j

The metric is independent of t. The lapse N and spatial metric γ_{ij}
are free functions of the spatial coordinates.

# Returns
A `MetricAnsatzResult` with 1 + 6 = 7 non-zero components (in 4D).
"""
function symmetry_reduce(sym::StaticSymmetry;
                          coords::Vector{Symbol}=[:t, :x, :y, :z])
    length(coords) == 4 ||
        error("symmetry_reduce(StaticSymmetry): need 4 coordinates")

    # g_{tt} = -N², g_{ti} = 0 (static: no cross-terms)
    # g_{ij} = γ_{ij} (6 independent components in 3D)
    components = Dict{Tuple{Int,Int}, Any}(
        (1, 1) => (:neg, :N_lapse),
    )
    # Spatial metric components
    spatial_fns = Symbol[]
    for i in 2:4
        for j in i:4
            fn = Symbol(:gamma_, coords[i], coords[j])
            components[(i, j)] = fn
            push!(spatial_fns, fn)
        end
    end

    MetricAnsatzResult(
        components, coords,
        [:N_lapse; spatial_fns],
        coords[2:4],  # spatial coordinates
        sym, 4
    )
end

# ── Homogeneous + Isotropic (FLRW) ──────────────────────────────────

"""
    symmetry_reduce(::HomogeneousIsotropy;
                     coords=[:t, :r, :theta, :phi],
                     k::Int=0) -> MetricAnsatzResult

Generate the FLRW metric:

    ds² = -dt² + a(t)² [dr²/(1-kr²) + r²dΩ²]

where a(t) is the scale factor and k = 0, ±1 is the spatial curvature.

# Returns
A `MetricAnsatzResult` with 4 non-zero components and 1 free function.
"""
function symmetry_reduce(sym::HomogeneousIsotropy;
                          coords::Vector{Symbol}=[:t, :r, :theta, :phi],
                          k::Int=0)
    length(coords) == 4 ||
        error("symmetry_reduce(HomogeneousIsotropy): need 4 coordinates")
    k in (-1, 0, 1) ||
        error("symmetry_reduce: k must be -1, 0, or 1, got $k")

    # g_{tt} = -1, g_{rr} = a(t)²/(1-kr²), g_{θθ} = a(t)²r², g_{φφ} = a(t)²r²sin²θ
    components = Dict{Tuple{Int,Int}, Any}(
        (1, 1) => -1,                     # -1 (exact)
        (2, 2) => (:a_squared_over_f, k), # a²/(1-kr²)
        (3, 3) => (:a_squared_r2,),       # a²r²
        (4, 4) => (:a_squared_r2_sin2,),  # a²r²sin²θ
    )

    MetricAnsatzResult(
        components, coords,
        [:a_scale],           # the scale factor a(t)
        [coords[1]],          # essential coordinate: t
        sym, 4
    )
end

# ── Axial Symmetry (Lewis-Papapetrou) ────────────────────────────────

"""
    symmetry_reduce(::AxialSymmetry;
                     coords=[:t, :r, :theta, :phi]) -> MetricAnsatzResult

Generate the stationary axisymmetric (Lewis-Papapetrou) metric:

    ds² = -N² dt² + g_{rr} dr² + g_{θθ} dθ² + g_{φφ}(dφ - ω dt)²

The metric is independent of t and φ. Free functions depend on (r, θ).

# Returns
A `MetricAnsatzResult` with 5 non-zero components and 5 free functions.
"""
function symmetry_reduce(sym::AxialSymmetry;
                          coords::Vector{Symbol}=[:t, :r, :theta, :phi])
    length(coords) == 4 ||
        error("symmetry_reduce(AxialSymmetry): need 4 coordinates")

    # Lewis-Papapetrou form: 5 free functions of (r,θ)
    components = Dict{Tuple{Int,Int}, Any}(
        (1, 1) => (:neg_N2_plus_omega2_gphph,), # g_{tt}
        (1, 4) => :omega_gphph,                  # g_{tφ} = ω g_{φφ}
        (2, 2) => :grr_LP,                       # g_{rr}
        (3, 3) => :gthth_LP,                     # g_{θθ}
        (4, 4) => :gphph_LP,                     # g_{φφ}
    )

    MetricAnsatzResult(
        components, coords,
        [:N_LP, :grr_LP, :gthth_LP, :gphph_LP, :omega_LP],
        [coords[2], coords[3]],  # essential: r, θ
        sym, 4
    )
end

# ── Utility functions ────────────────────────────────────────────────

"""
    independent_components(result::MetricAnsatzResult) -> Int

Number of independent metric components (free functions).
"""
independent_components(result::MetricAnsatzResult) = length(result.free_functions)

"""
    constrained_components(result::MetricAnsatzResult) -> Int

Number of metric components fixed to zero or constant by the symmetry.
Total d(d+1)/2 minus the independent components.
"""
function constrained_components(result::MetricAnsatzResult)
    total = result.dim * (result.dim + 1) ÷ 2
    total - length(result.components)
end
