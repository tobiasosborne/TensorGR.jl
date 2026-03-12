#= Geodesic equation for numerical integration.

Provides the GeodesicEquation type and ODE right-hand side for integrating
geodesics on a given spacetime. Designed for use with DifferentialEquations.jl:

    prob = ODEProblem(geodesic_rhs!, u0, tspan, geq)
    sol = solve(prob, ...)

where geq::GeodesicEquation is constructed via setup_geodesic.
=#

"""
    GeodesicEquation

Stores the data needed to evaluate the geodesic ODE right-hand side.

# Fields
- `metric_fn` -- callable `(x::AbstractVector) -> (g::Matrix, ginv::Matrix)`
  returning the metric and its inverse at a spacetime point.
- `christoffel_fn` -- callable `(x::AbstractVector) -> Gamma::Array{Float64,3}`
  returning Christoffel symbols at a spacetime point (`Gamma[a,b,c]` = Gamma^a_{bc}).
- `dim::Int` -- spacetime dimension.
- `is_timelike::Bool` -- `true` for timelike geodesics, `false` for null/spacelike.
"""
struct GeodesicEquation
    metric_fn::Any          # (x) -> (g, ginv)
    christoffel_fn::Any     # (x) -> Gamma[a,b,c]
    dim::Int
    is_timelike::Bool
end

"""
    _numerical_christoffel(metric_fn, x, dim; epsilon=1e-8)

Compute Christoffel symbols at point `x` using central finite differences
for the metric derivatives.

Returns `Gamma::Array{Float64,3}` with `Gamma[a,b,c]` = Gamma^a_{bc}.
"""
function _numerical_christoffel(metric_fn, x::AbstractVector, dim::Int;
                                 epsilon::Float64=1e-8)
    g0, ginv0 = metric_fn(x)

    # Compute metric derivatives via central finite differences:
    # dg[i,j,k] = partial_k g_{ij}
    dg = Array{Float64}(undef, dim, dim, dim)
    for k in 1:dim
        xp = copy(x)
        xm = copy(x)
        xp[k] += epsilon
        xm[k] -= epsilon
        gp, _ = metric_fn(xp)
        gm, _ = metric_fn(xm)
        for i in 1:dim, j in 1:dim
            dg[i, j, k] = (gp[i, j] - gm[i, j]) / (2 * epsilon)
        end
    end

    # Christoffel: Gamma^a_{bc} = (1/2) g^{ad} (partial_b g_{cd} + partial_c g_{bd} - partial_d g_{bc})
    Gamma = Array{Float64}(undef, dim, dim, dim)
    for a in 1:dim, b in 1:dim, c in 1:dim
        s = 0.0
        for d in 1:dim
            s += ginv0[a, d] * (dg[c, d, b] + dg[b, d, c] - dg[b, c, d])
        end
        Gamma[a, b, c] = s / 2
    end

    Gamma
end

"""
    setup_geodesic(metric_fn; dim, is_timelike=true, christoffel_fn=nothing)

Construct a `GeodesicEquation` for numerical integration.

# Arguments
- `metric_fn` -- callable `(x::AbstractVector) -> (g::Matrix, ginv::Matrix)`.
- `dim::Int` -- spacetime dimension.
- `is_timelike::Bool` -- whether geodesics are timelike (default `true`).
- `christoffel_fn` -- optional callable `(x::AbstractVector) -> Gamma::Array{Float64,3}`.
  If not provided, Christoffel symbols are computed numerically from `metric_fn`
  via central finite differences at each evaluation point.

# Returns
A `GeodesicEquation` ready for use with `geodesic_rhs!`.

# Example
```julia
# Minkowski metric in 4D
function mink(x)
    g = diagm([-1.0, 1.0, 1.0, 1.0])
    ginv = diagm([-1.0, 1.0, 1.0, 1.0])
    (g, ginv)
end
geq = setup_geodesic(mink; dim=4)
```
"""
function setup_geodesic(metric_fn;
                        dim::Int,
                        is_timelike::Bool=true,
                        christoffel_fn=nothing)
    if christoffel_fn === nothing
        christoffel_fn = x -> _numerical_christoffel(metric_fn, x, dim)
    end
    GeodesicEquation(metric_fn, christoffel_fn, dim, is_timelike)
end

"""
    geodesic_rhs!(du, u, p::GeodesicEquation, tau)

In-place ODE right-hand side for the geodesic equation, compatible with
DifferentialEquations.jl.

The state vector `u` has length `2*dim`:
- `u[1:dim]` = position x^mu
- `u[dim+1:2*dim]` = velocity dx^mu/dtau

The derivative `du` is filled as:
- `du[1:dim]` = velocity (dx^mu/dtau = v^mu)
- `du[dim+1:2*dim]` = acceleration (dv^mu/dtau = -Gamma^mu_{alpha beta} v^alpha v^beta)

The parameter `p` must be a `GeodesicEquation`.
"""
function geodesic_rhs!(du, u, p::GeodesicEquation, tau)
    dim = p.dim
    x = @view u[1:dim]
    v = @view u[dim+1:2*dim]

    # Evaluate Christoffel symbols at current position
    Gamma = p.christoffel_fn(x)

    # Velocities: dx^mu/dtau = v^mu
    for mu in 1:dim
        du[mu] = v[mu]
    end

    # Accelerations: dv^mu/dtau = -Gamma^mu_{alpha,beta} v^alpha v^beta
    for mu in 1:dim
        acc = 0.0
        for alpha in 1:dim, beta in 1:dim
            acc += Gamma[mu, alpha, beta] * v[alpha] * v[beta]
        end
        du[dim + mu] = -acc
    end

    nothing
end
